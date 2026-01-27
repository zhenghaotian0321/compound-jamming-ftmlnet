import os
import re
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout,
    BatchNormalization, Concatenate, GlobalAveragePooling2D,
    Multiply, Reshape, GlobalMaxPooling2D, UpSampling2D,
    GlobalAveragePooling1D, Lambda, Layer
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt  # 新增：用于绘制混淆矩阵

# ====== 参数 ======
img_size = (64, 64)
batch_size = 32
num_jam_types = 4  # 0:FSK,1:BPSK,2:QPSK,3:16QAM
train_data_dir = r'E:\xxbDigitalJamTrainset1000'
test_data_dir = r'E:\xxbDigitalJamTestset5000'

# 先验概率顺序对应 [FSK,BPSK,QPSK,16QAM]
PRIOR_VECTOR = tf.constant([0.25, 0.25, 0.25, 0.25], dtype=tf.float32)


# ========== 随机偏移裁剪（TF 原生实现） ==========
def random_shift_crop_tf(image, max_shift=0.3):
    """TF 原生随机垂直偏移裁剪 + 填充，保持尺寸不变"""
    h = tf.shape(image)[0]
    max_shift_px = tf.cast(tf.math.floor(tf.cast(h, tf.float32) * max_shift), tf.int32)
    max_shift_px = tf.maximum(0, tf.minimum(max_shift_px, h - 1))

    def no_op():
        return image

    def do_shift():
        # 随机选择 1..max_shift_px
        shift_px = tf.random.uniform([], minval=1, maxval=max_shift_px + 1, dtype=tf.int32)
        # 随机方向
        do_up = tf.less(tf.random.uniform([], 0.0, 1.0), 0.5)

        def shift_up():
            cropped = image[shift_px:, :, :]
            pad_h = tf.shape(image)[0] - tf.shape(cropped)[0]
            padded = tf.pad(cropped, [[0, pad_h], [0, 0], [0, 0]])
            return padded

        def shift_down():
            cropped = image[:-shift_px, :, :]
            pad_h = tf.shape(image)[0] - tf.shape(cropped)[0]
            padded = tf.pad(cropped, [[pad_h, 0], [0, 0], [0, 0]])
            return padded

        return tf.cond(do_up, shift_up, shift_down)

    return tf.cond(max_shift_px > 0, do_shift, no_op)


# ========== 空间金字塔池化（修复版） ==========
class SpatialPyramidPooling(Layer):
    def __init__(self, **kwargs):
        super(SpatialPyramidPooling, self).__init__(**kwargs)

    def call(self, inputs):
        # 使用固定的上采样倍数而不是动态尺寸
        # 4x4池化 + 4倍上采样
        pool1 = MaxPooling2D(pool_size=(4, 4))(inputs)
        pool1 = UpSampling2D(size=(4, 4), interpolation='bilinear')(pool1)

        # 2x2池化 + 2倍上采样
        pool2 = MaxPooling2D(pool_size=(2, 2))(inputs)
        pool2 = UpSampling2D(size=(2, 2), interpolation='bilinear')(pool2)

        # 全局平均池化 + 使用Lambda层进行动态上采样
        pool3 = GlobalAveragePooling2D(keepdims=True)(inputs)

        # 使用tf.image.resize进行动态尺寸上采样
        def resize_to_input(x):
            input_tensor, pool_tensor = x
            target_shape = tf.shape(input_tensor)[1:3]  # 获取H, W
            return tf.image.resize(pool_tensor, target_shape, method='bilinear')

        pool3 = Lambda(resize_to_input)([inputs, pool3])

        # 合并所有特征
        merged = Concatenate(axis=-1)([inputs, pool1, pool2, pool3])
        return merged

    def compute_output_shape(self, input_shape):
        # 输出通道数 = 输入通道数 * 4 (原始 + 3个池化分支)
        return (input_shape[0], input_shape[1], input_shape[2], input_shape[3] * 4)


# ========== 统计特征计算（TF） ==========
def compute_statistical_features(image):
    # image: [H,W,3] float32 in [0,1]
    gray = tf.image.rgb_to_grayscale(image)  # (H,W,1)
    mean = tf.reduce_mean(gray)
    variance = tf.math.reduce_variance(gray)
    std_dev = tf.math.sqrt(variance + 1e-7)
    centered = gray - mean
    skewness = tf.reduce_mean(tf.pow(centered / (std_dev + 1e-7), 3))
    kurtosis = tf.reduce_mean(tf.pow(centered / (std_dev + 1e-7), 4)) - 3.0
    energy = tf.reduce_mean(tf.square(gray))
    gray_flat = tf.reshape(gray, [-1])
    hist = tf.histogram_fixed_width(gray_flat, [0.0, 1.0], nbins=64)
    hist = tf.cast(hist, tf.float32)
    hist_sum = tf.reduce_sum(hist)
    hist_norm = hist / (hist_sum + 1e-7)
    entropy = -tf.reduce_sum(hist_norm * tf.math.log(hist_norm + 1e-7))
    min_val = tf.reduce_min(gray)
    max_val = tf.reduce_max(gray)
    peak_to_peak = max_val - min_val
    features = tf.stack([mean, variance, skewness, kurtosis, energy, entropy, peak_to_peak])
    features = tf.reshape(features, (7,))
    return features


# ========== 从文件名解析标签（TF 原生，不用 py_function） ==========
def parse_label_from_filename(file_path):
    """
    file_path: scalar string tensor (full path)
    返回 multi-hot vector shape (num_jam_types,)
    """
    filename = tf.strings.split(file_path, os.sep)[-1]

    # default zeros
    zero_vec = tf.zeros([num_jam_types], dtype=tf.float32)

    # case SINGLE: format like ...SINGLE_{TYPE}_...
    is_single = tf.strings.regex_full_match(filename, '.*SINGLE_.*')

    def handle_single():
        parts = tf.strings.split(filename, '_')
        # parts[1] should be type like FSK/BPSK/QPSK/16QAM
        type_str = parts[1]
        # simpler: compare strings
        conds = [
            tf.equal(type_str, 'FSK'),
            tf.equal(type_str, 'BPSK'),
            tf.equal(type_str, 'QPSK'),
            tf.equal(type_str, '16QAM')
        ]
        vecs = [
            tf.constant([1.0, 0.0, 0.0, 0.0], dtype=tf.float32),
            tf.constant([0.0, 1.0, 0.0, 0.0], dtype=tf.float32),
            tf.constant([0.0, 0.0, 1.0, 0.0], dtype=tf.float32),
            tf.constant([0.0, 0.0, 0.0, 1.0], dtype=tf.float32)
        ]
        # default zero
        out = zero_vec
        for c, v in zip(conds, vecs):
            out = tf.where(c, v, out)
        return out

    def handle_mixed():
        # expect pattern ...Types(\d+)_...
        has_types = tf.strings.regex_full_match(filename, '.*Types[0-9]+_.*')

        def extract_types_digits():
            # use regex_replace to keep only digits between Types and _
            digits = tf.strings.regex_replace(filename, '.*Types([0-9]+)_.*', '\\1')
            # split digits into characters
            chars = tf.strings.bytes_split(digits)  # e.g. ['1','2']
            # convert to numbers and map to indices (digit-1)
            nums = tf.strings.to_number(chars, out_type=tf.int32)  # e.g. [1,2]
            idxs = nums - 1
            # clip valid range
            idxs = tf.boolean_mask(idxs, tf.logical_and(idxs >= 0, idxs < num_jam_types))
            # make one-hot and sum
            onehots = tf.one_hot(idxs, depth=num_jam_types, dtype=tf.float32)
            if tf.size(onehots) == 0:
                return zero_vec
            summed = tf.reduce_sum(onehots, axis=0)
            # ensure binary (0/1)
            summed = tf.clip_by_value(summed, 0.0, 1.0)
            return summed

        return tf.cond(has_types, extract_types_digits, lambda: zero_vec)

    label_vec = tf.cond(is_single, handle_single, handle_mixed)
    label_vec.set_shape([num_jam_types])
    return label_vec


# ========== 读取并解析图像（TF） ==========
def _read_and_preprocess(file_path, is_training=False):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, img_size)
    img = tf.image.convert_image_dtype(img, tf.float32)
    if is_training:
        img = random_shift_crop_tf(img, max_shift=0.3)
        # ensure resized back to exact img_size
        img = tf.image.resize(img, img_size)
    label = parse_label_from_filename(file_path)
    return img, label


# ========== 创建基础 dataset ==========
def create_custom_dataset(data_dir, is_training=False):
    file_pattern = os.path.join(data_dir, '*.png')
    files = tf.data.Dataset.list_files(file_pattern, shuffle=is_training)
    ds = files.map(lambda p: _read_and_preprocess(p, is_training), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# ========== Helper: 生成 prior 和 auxiliary labels（TF，batch 级） ==========
def batch_generate_prior_and_aux(batch_labels):
    # batch_labels: (B, C) float32
    # prior_probs: for each sample, set prior for labels present (labels * PRIOR_VECTOR)
    prior_probs = batch_labels * PRIOR_VECTOR  # (B,C)

    # mod_types: 0 if label 0 or 1 present else 1 if label 2 or 3 present
    cond_mod0 = tf.logical_or(batch_labels[:, 0] > 0.5, batch_labels[:, 1] > 0.5)
    cond_mod1 = tf.logical_or(batch_labels[:, 2] > 0.5, batch_labels[:, 3] > 0.5)
    mod_types = tf.where(cond_mod0, tf.zeros_like(batch_labels[:, 0], dtype=tf.int32),
                         tf.where(cond_mod1, tf.ones_like(batch_labels[:, 0], dtype=tf.int32),
                                  tf.zeros_like(batch_labels[:, 0], dtype=tf.int32)))
    # stats_types: 0 if FSK present else 1
    stats_types = tf.where(batch_labels[:, 0] > 0.5, tf.zeros_like(batch_labels[:, 0], dtype=tf.int32),
                           tf.ones_like(batch_labels[:, 0], dtype=tf.int32))
    return prior_probs, mod_types, stats_types


# ========== create_enhanced_dataset: 输出为 ((inputs), (outputs)) ==========
def create_enhanced_dataset(data_dir, is_training=False):
    base_ds = create_custom_dataset(data_dir, is_training=is_training)

    def add_features(batch_images, batch_labels):
        # batch_images: (B,H,W,3), batch_labels: (B,C)
        stats_features = tf.map_fn(lambda im: compute_statistical_features(im),
                                   batch_images,
                                   fn_output_signature=tf.float32)
        # stats_features shape (B,7)
        prior_probs, mod_labels, stats_labels = tf.numpy_function(
            lambda lbls: (batch_generate_prior_and_aux(lbls)[0].numpy(),
                          batch_generate_prior_and_aux(lbls)[1].numpy(),
                          batch_generate_prior_and_aux(lbls)[2].numpy()),
            [batch_labels],
            (tf.float32, tf.int32, tf.int32)
        )
        # The above uses numpy_function to assemble batch-level ops. Could be implemented fully in TF,
        # but for simplicity we use numpy_function for the batch conversion and then set shapes.
        prior_probs = tf.convert_to_tensor(prior_probs, dtype=tf.float32)
        mod_labels = tf.convert_to_tensor(mod_labels, dtype=tf.int32)
        stats_labels = tf.convert_to_tensor(stats_labels, dtype=tf.int32)

        prior_probs.set_shape([None, num_jam_types])
        mod_labels.set_shape([None])
        stats_labels.set_shape([None])
        stats_features.set_shape([None, 7])

        inputs = (batch_images, prior_probs, stats_features, batch_labels)
        outputs = (batch_labels, mod_labels, stats_labels)
        return inputs, outputs

    ds = base_ds.map(add_features, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# ========== Center + Prototype Loss 层 (TF2 兼容修复版) ==========
class CenterProtoLossLayer(tf.keras.layers.Layer):
    """
    维护类中心并在 call 中:
      - 计算 center_loss 和 proto_loss 并 add_loss
      - 在 training=True 时更新 centers (self.centers.assign)
    输入: (embeddings, labels) embeddings: (B,D), labels: (B,C) multi-hot float32
    """

    def __init__(self, num_classes, embed_dim, alpha=0.5, lambda_center=0.003, lambda_proto=1.0, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.alpha = alpha
        self.lambda_center = lambda_center
        self.lambda_proto = lambda_proto

    def build(self, input_shape):
        # centers shape: (num_classes, embed_dim)
        self.centers = self.add_weight(
            name='centers',
            shape=(self.num_classes, self.embed_dim),
            initializer='zeros',
            trainable=False,
            dtype=tf.float32
        )
        super().build(input_shape)

    def call(self, inputs, training=None):
        embeddings, labels = inputs
        labels = tf.cast(labels, tf.float32)  # (B,C)
        batch_size = tf.shape(embeddings)[0]

        # Distances (B,C)
        x2 = tf.reduce_sum(tf.square(embeddings), axis=1, keepdims=True)  # (B,1)
        c2 = tf.reduce_sum(tf.square(self.centers), axis=1)  # (C,)
        cross = tf.matmul(embeddings, self.centers, transpose_b=True)  # (B,C)
        dists = x2 - 2.0 * cross + c2  # (B,C)

        # Center loss (only positive labels)
        masked = dists * labels
        center_loss = tf.reduce_sum(masked) / (tf.reduce_sum(labels) + 1e-7)

        # Proto loss: BCE on sigmoid(-dist)
        logits = -dists
        proto_probs = tf.math.sigmoid(logits)
        bce = tf.keras.losses.binary_crossentropy(labels, proto_probs)
        proto_loss = tf.reduce_mean(bce)

        total_loss = self.lambda_center * center_loss + self.lambda_proto * proto_loss
        self.add_loss(total_loss)

        # Update centers only during training
        if training:
            labels_T = tf.transpose(labels)  # (C,B)
            sum_embeddings_per_class = tf.matmul(labels_T, embeddings)  # (C,D)
            class_counts = tf.reduce_sum(labels_T, axis=1)  # (C,)

            counts_expand = tf.expand_dims(class_counts, axis=1)  # (C,1)
            mean_emb_per_class = sum_embeddings_per_class / (counts_expand + 1e-7)  # (C,D)

            have_class = tf.expand_dims(tf.cast(class_counts > 0.0, tf.float32), axis=1)  # (C,1)
            delta = (mean_emb_per_class - self.centers) * have_class  # (C,D) only for present classes
            new_centers = self.centers + self.alpha * delta

            # TF2 正确更新方式
            self.centers.assign(new_centers)

        # 返回 embeddings 不变（仅用于 loss/centers）
        return embeddings

    def compute_output_shape(self, input_shape):
        # 输入 embeddings 的 shape 就是输出 shape
        return input_shape[0]


# ========== Channel Attention (保持) ==========
def channel_attention(input_feature, ratio=8):
    channel = int(input_feature.shape[-1])
    shared_layer_one = Dense(channel // ratio,
                             activation='relu',
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    shared_layer_two = Dense(channel,
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    avg_pool = GlobalAveragePooling2D()(input_feature)
    avg_pool = Reshape((1, 1, channel))(avg_pool)
    avg_pool = shared_layer_one(avg_pool)
    avg_pool = shared_layer_two(avg_pool)
    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1, 1, channel))(max_pool)
    max_pool = shared_layer_one(max_pool)
    max_pool = shared_layer_two(max_pool)
    cbam_feature = tf.keras.layers.Add()([avg_pool, max_pool])
    cbam_feature = tf.keras.layers.Activation('sigmoid')(cbam_feature)
    return Multiply()([input_feature, cbam_feature])


# ========== 构建模型（包含 embedding 和 CenterProtoLossLayer） ==========
def build_enhanced_mpaml_model_multi_label_with_proto():
    img_input = Input(shape=(img_size[0], img_size[1], 3), name='image_input')
    prior_input = Input(shape=(num_jam_types,), name='prior_input')
    stats_input = Input(shape=(7,), name='stats_input')
    labels_input = Input(shape=(num_jam_types,), name='labels_input')  # multi-hot labels as input

    x = Conv2D(32, (3, 3), activation='relu', padding='same')(img_input)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.2)(x)

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.3)(x)

    # 使用自定义层替代原来的函数
    x = SpatialPyramidPooling()(x)

    # mod branch
    mod_branch = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    mod_branch = BatchNormalization()(mod_branch)
    mod_branch = channel_attention(mod_branch)
    mod_branch = MaxPooling2D((2, 2))(mod_branch)
    mod_branch = Flatten()(mod_branch)

    # energy branch
    energy_branch = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    energy_branch = BatchNormalization()(energy_branch)
    energy_branch = channel_attention(energy_branch)
    energy_branch = MaxPooling2D((2, 2))(energy_branch)
    energy_branch = Flatten()(energy_branch)

    # stats branch
    stats_branch = Dense(64, activation='relu')(stats_input)
    stats_branch = BatchNormalization()(stats_branch)
    stats_branch = Dropout(0.3)(stats_branch)
    stats_branch = Dense(32, activation='relu')(stats_branch)

    # auxiliary tasks
    mod_task = Dense(64, activation='relu')(mod_branch)
    mod_task = Dropout(0.3)(mod_task)
    mod_output = Dense(2, activation='softmax', name='mod_type')(mod_task)
    stats_output = Dense(2, activation='softmax', name='stats_type')(stats_branch)

    img_feat = Concatenate()([mod_branch, energy_branch, mod_task])
    img_feat = Dense(128, activation='relu')(img_feat)
    combined = Concatenate()([img_feat, stats_branch, prior_input])

    x_dense = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(combined)
    x_dense = Dropout(0.5)(x_dense)
    x_dense = Dense(128, activation='relu')(x_dense)
    x_dense = Dropout(0.4)(x_dense)

    embedding_dim = 64
    embed = Dense(embedding_dim, activation=None, name='embedding')(x_dense)
    embed_norm = Lambda(lambda z: tf.nn.l2_normalize(z, axis=1), name='embed_norm')(embed)

    center_proto_layer = CenterProtoLossLayer(num_classes=num_jam_types,
                                              embed_dim=embedding_dim,
                                              alpha=0.5,
                                              lambda_center=0.003,
                                              lambda_proto=1.0,
                                              name='center_proto')
    # call to register loss and update centers (during training)
    _ = center_proto_layer((embed_norm, labels_input))

    main_output = Dense(num_jam_types, activation='sigmoid', name='main_output')(x_dense)

    model = Model(inputs=[img_input, prior_input, stats_input, labels_input],
                  outputs=[main_output, mod_output, stats_output])
    return model


# ========== 创建数据集并检查混合样本统计 ==========
train_ds = create_enhanced_dataset(train_data_dir, is_training=True)
test_ds = create_enhanced_dataset(test_data_dir, is_training=False)


def check_mixed_samples(dataset, name='dataset'):
    mixed_count = 0
    total_count = 0
    for batch in dataset:
        inputs, outputs = batch
        labels = inputs[3].numpy()
        for label_vec in labels:
            num_labels = np.sum(label_vec > 0.5)
            if num_labels > 1:
                mixed_count += 1
            total_count += 1
    print(f"{name} 统计: 总样本={total_count}, 混合样本={mixed_count} ({mixed_count / total_count * 100:.1f}%)")


print("训练集样本统计:")
check_mixed_samples(train_ds, name='train_ds')
print("\n测试集样本统计:")
check_mixed_samples(test_ds, name='test_ds')

# ========== 构建和编译模型 ==========
model = build_enhanced_mpaml_model_multi_label_with_proto()

loss_weights = {
    'main_output': 1.0,
    'mod_type': 0.7,
    'stats_type': 0.8
}

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss={
        'main_output': 'binary_crossentropy',
        'mod_type': 'sparse_categorical_crossentropy',
        'stats_type': 'sparse_categorical_crossentropy'
    },
    loss_weights=loss_weights,
    metrics={
        'main_output': 'binary_accuracy',
        'mod_type': 'accuracy',
        'stats_type': 'accuracy'
    }
)

# ========== callbacks ==========
early_stop = EarlyStopping(
    monitor='val_main_output_binary_accuracy',
    mode='max',
    patience=1000,
    restore_best_weights=True,
    verbose=1
)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    mode='min',
    factor=0.5,
    patience=1000,
    min_lr=1e-6,
    verbose=1
)

# ========== 训练 ==========
history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=1000,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)


# ========== 评估（预测时传入 labels 占位） ==========
def evaluate_multi_label_model(model, dataset):
    jam_type_names = ['FSK', 'BPSK', 'QPSK', '16QAM']
    results = {}
    for jam in jam_type_names:
        results[jam] = {'single': {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0, 'samples': 0},
                        'mixed': {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0, 'samples': 0},
                        'total': {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0, 'samples': 0}}
    results['total_samples'] = 0
    results['mixed_samples'] = 0
    results['single_samples'] = 0
    jam_idx = {'FSK': 0, 'BPSK': 1, 'QPSK': 2, '16QAM': 3}

    for batch in dataset:
        inputs, outputs = batch
        images, priors, stats, true_labels_input = inputs
        true_labels, _, _ = outputs

        zeros_labels = np.zeros((images.shape[0], num_jam_types), dtype=np.float32)
        preds = model.predict([images, priors, stats, zeros_labels], verbose=0)
        pred_labels = (preds[0] > 0.5).astype(int)

        true_labels = true_labels.numpy()
        for i in range(len(true_labels)):
            true_vec = true_labels[i]
            pred_vec = pred_labels[i]
            num_true = np.sum(true_vec)
            is_single = num_true == 1
            sample_type = 'single' if is_single else 'mixed'
            if is_single:
                results['single_samples'] += 1
            else:
                results['mixed_samples'] += 1
            for jam_name, idx in jam_idx.items():
                true_val = true_vec[idx]
                pred_val = pred_vec[idx]
                if true_val > 0.5:
                    results[jam_name][sample_type]['samples'] += 1
                    results[jam_name]['total']['samples'] += 1
                if true_val > 0.5:
                    if pred_val > 0.5:
                        results[jam_name][sample_type]['TP'] += 1
                        results[jam_name]['total']['TP'] += 1
                    else:
                        results[jam_name][sample_type]['FN'] += 1
                        results[jam_name]['total']['FN'] += 1
                else:
                    if pred_val > 0.5:
                        results[jam_name][sample_type]['FP'] += 1
                        results[jam_name]['total']['FP'] += 1
                    else:
                        results[jam_name][sample_type]['TN'] += 1
                        results[jam_name]['total']['TN'] += 1
            results['total_samples'] += 1

    for jam_name in jam_idx.keys():
        for t in ['single', 'mixed', 'total']:
            st = results[jam_name][t]
            TP, FP, FN, TN = st['TP'], st['FP'], st['FN'], st['TN']

            total = TP + FP + FN + TN
            # 准确率
            st['accuracy'] = (TP + TN) / total if total > 0 else None

            # 精确率 Precision = TP / (TP + FP)
            prec_den = TP + FP
            st['precision'] = TP / prec_den if prec_den > 0 else 0.0  # 如果分母为0，设为0

            # 召回率 Recall = TP / (TP + FN)
            rec_den = TP + FN
            st['recall'] = TP / rec_den if rec_den > 0 else 0.0  # 如果分母为0，设为0

            # F1 = 2 * P * R / (P + R)
            if st['precision'] is not None and st['recall'] is not None:
                denom = st['precision'] + st['recall']
                st['f1'] = 2 * st['precision'] * st['recall'] / denom if denom > 0 else 0.0
            else:
                st['f1'] = 0.0

    return results


results = evaluate_multi_label_model(model, test_ds)
print("\n===== 报告 =====")
print(f"总样本数: {results['total_samples']}")
print(f"单类样本: {results['single_samples']}")
print(f"混合样本: {results['mixed_samples']}")

for jam in ['FSK', 'BPSK', 'QPSK', '16QAM']:
    print(f"\n--- {jam} ---")

    def print_stats(tag, st):
        if st['samples'] == 0:
            print(f"{tag}：无样本")
        else:
            # 使用安全的格式化，确保值不为None
            acc_str = f"{st['accuracy']:.4f}" if st['accuracy'] is not None else "N/A"
            prec_str = f"{st['precision']:.4f}"
            rec_str = f"{st['recall']:.4f}"
            f1_str = f"{st['f1']:.4f}"

            print(
                f"{tag}：出现 {st['samples']}，"
                f"Acc={acc_str}, "
                f"P={prec_str}, "
                f"R={rec_str}, "
                f"F1={f1_str}"
            )

    print_stats("单类", results[jam]['single'])
    print_stats("混合", results[jam]['mixed'])
    print_stats("总体", results[jam]['total'])


# ========== 绘制每个干扰类型的 2x2 混淆矩阵（包含单类+混合） ==========
def plot_per_class_confusion_matrices(results):
    """
    使用 'total' 统计（即 single+mixed 全部样本），
    对每个干扰类型构造 2x2 混淆矩阵：
        [[TN, FP],
         [FN, TP]]
    然后画成 4 个子图。
    """
    jam_type_names = ['FSK', 'BPSK', 'QPSK', '16QAM']
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))

    for idx, jam in enumerate(jam_type_names):
        st = results[jam]['total']
        TP, FP, FN, TN = st['TP'], st['FP'], st['FN'], st['TN']
        cm = np.array([[TN, FP],
                       [FN, TP]], dtype=np.int32)

        ax = axes[idx]
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.set_title(f"{jam}\nAcc={st['accuracy']:.3f} P={st['precision']:.3f}\n"
                     f"R={st['recall']:.3f} F1={st['f1']:.3f}")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Pred 0', 'Pred 1'])
        ax.set_yticklabels(['True 0', 'True 1'])

        # 在格子中写数字
        thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
        for i in range(2):
            for j in range(2):
                color = "white" if cm[i, j] > thresh else "black"
                ax.text(j, i, str(cm[i, j]),
                        horizontalalignment="center",
                        verticalalignment="center",
                        color=color)

    fig.tight_layout()
    plt.show()


plot_per_class_confusion_matrices(results)
