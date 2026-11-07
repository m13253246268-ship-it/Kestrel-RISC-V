/* ================================================================
 * qk_repack.c - 跨平台标量 repack（converter x86 路径）
 *
 * 布局与 src/model/vllm_safetensors.c 的 NEON 分支逐字节一致
 * （纯字节搬移，无浮点；q4_0_4x4/q8_0_8x8/q8_0_4x4 逻辑对应引擎
 * 行 3369-3407 / 3427-3477 / 3529-3562；q4_0_8x8l 对应行 3494-3520）。
 * 与引擎差异：临时组缓冲用 malloc（引擎用预分配 scratch 全局），
 * 几何/返回语义（不可 repack -> -1，buffer 不动）保持一致。
 * 对拍红线：板端 NEON 引擎同函数 vs 本文件标量，同输入哈希一致。
 * ================================================================ */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ---------------- Q4_0 legacy -> 4x4 (nibble) ----------------
 * per 4-row group: d[4] f16 scales (row order), then 16 interleaved
 * 4-byte chunks, each ^0x88888888. Output = 72 B per 32-col block. */
int repack_q4_0_4x4_inplace(uint8_t *buf, int rows, int cols) {
    if (rows % 4 != 0 || cols % 32 != 0) return -1;
    int nb = cols / 32;
    size_t row_stride = (size_t)nb * 18;
    size_t group_bytes = 4 * row_stride;
    uint8_t *tmp = (uint8_t *)malloc(group_bytes);
    if (!tmp) return -1;
    for (int r0 = 0; r0 < rows; r0 += 4) {
        uint8_t *dst = buf + (size_t)(r0 >> 2) * group_bytes;
        memcpy(tmp, dst, group_bytes);
        for (int b = 0; b < nb; b++) {
            uint8_t *out = dst + (size_t)b * 72;
            for (int i = 0; i < 4; i++)
                memcpy(out + (size_t)i * 2,
                       tmp + (size_t)i * row_stride + (size_t)b * 18, 2);
            const uint32_t xor_mask = 0x88888888u;
            for (int i = 0; i < 16; i++) {
                uint32_t e;
                memcpy(&e, tmp + (size_t)(i & 3) * row_stride + (size_t)b * 18 + 2
                             + (size_t)(i >> 2) * 4, 4);
                e ^= xor_mask;
                memcpy(out + 8 + (size_t)i * 4, &e, 4);
            }
        }
    }
    free(tmp);
    return 0;
}

/* ---------------- Q4_0 legacy -> lane-major (32-row tile) ----------------
 * 32 行 tile（8 个 4 行组），每块（32 列）：
 *   scale: 64B = 32 个 f16，顺序 [m][g]（m=组内行 0..3, g=组 0..7）
 *   data : 512B = 16 字（i=k*4+m）× 8 组，顺序 [i][g]，字 ^0x88888888
 * 总 576B/块 = 8 组 × 72B，字节保持。g 最内层连续 → RVV vle32 一次读 8 组。 */
int repack_q4_0_lane_inplace(uint8_t *buf, int rows, int cols) {
    if (rows % 32 != 0 || cols % 32 != 0) return -1;
    int nb = cols / 32;
    int ntiles = rows / 32;
    size_t row_stride = (size_t)nb * 18;
    size_t tile_bytes = (size_t)nb * 576;
    uint8_t *tmp = (uint8_t *)malloc(tile_bytes);
    if (!tmp) return -1;
    for (int t = 0; t < ntiles; t++) {
        uint8_t *dst = buf + (size_t)t * tile_bytes;
        memcpy(tmp, dst, tile_bytes);
        for (int b = 0; b < nb; b++) {
            uint8_t *out = dst + (size_t)b * 576;
            /* scale: [m][g]，来源 legacy 行 (g*4+m) 块 b 的 [0,2) */
            for (int m = 0; m < 4; m++)
                for (int g = 0; g < 8; g++)
                    memcpy(out + (size_t)(m * 8 + g) * 2,
                           tmp + (size_t)(g * 4 + m) * row_stride + (size_t)b * 18, 2);
            /* data: [i][g]，i=k*4+m，来源 legacy 行 (g*4+m) 块 b 的 nibble 第 k 组 */
            const uint32_t xm = 0x88888888u;
            for (int i = 0; i < 16; i++) {
                int m = i & 3, k = i >> 2;
                for (int g = 0; g < 8; g++) {
                    uint32_t e;
                    memcpy(&e, tmp + (size_t)(g * 4 + m) * row_stride
                                 + (size_t)b * 18 + 2 + (size_t)k * 4, 4);
                    e ^= xm;
                    memcpy(out + 64 + (size_t)(i * 8 + g) * 4, &e, 4);
                }
            }
        }
    }
    free(tmp);
    return 0;
}

/* ---------------- Q4_0 legacy -> col-major 4-bit (T4-1 列主序) ----------------
 * 每 32 行 tile、每 block 576 B：
 *   [0,64)  32 个 f16 scale，**按行线性**：out[r*2] = 行 r 的块 b scale（r=0..31）；
 *   [64,576) 列主序 4-bit：列 c 占 16 B（out[64+c*16 + r/2]），行 r 的 nibble 放
 *            低半字节（r 偶）/ 高半字节（r 奇）。nibble 值 = legacy 原始 4-bit（即
 *            value+8），内核用 (nib & 0xF) - 8 还原有符号 int4，**不再异或 0x88**。
 * 与 lane(2) 同源（legacy 18B/块），仅重排；体积不变（576 B/块）。
 *
 * [SHS 公理映射: bitwise_5for_5fsuperposition_358d7499 (bitwise_or_superposition)]
 * 偶行低半字节 [3:0] 与奇行高半字节 [7:4] 位域正交不交（mask_lo & mask_hi == 0），
 * 直接成对合成单字节覆写，消除 memset(512B) 与 1024 次读-改-写 (|=) 冒险。 */
int repack_q4_0_lane_col_inplace(uint8_t *buf, int rows, int cols) {
    if (rows % 32 != 0 || cols % 32 != 0) return -1;
    int nb = cols / 32;
    int ntiles = rows / 32;
    size_t row_stride = (size_t)nb * 18;
    size_t tile_bytes = (size_t)nb * 576;   /* = 32 * row_stride */
    uint8_t *tmp = (uint8_t *)malloc(tile_bytes);
    if (!tmp) return -1;
    for (int t = 0; t < ntiles; t++) {
        uint8_t *dst = buf + (size_t)t * tile_bytes;
        memcpy(tmp, dst, tile_bytes);
        for (int b = 0; b < nb; b++) {
            uint8_t *out = dst + (size_t)b * 576;
            const uint8_t *blk_base = tmp + (size_t)b * 18;
            /* scale：按行线性（r*2） */
            for (int r = 0; r < 32; r++)
                memcpy(out + (size_t)r * 2, blk_base + (size_t)r * row_stride, 2);
            /* data：列主序，偶行低 / 奇行高正交合并直写（c=0..15 低半字节，c=16..31 高半字节） */
            for (int byteidx = 0; byteidx < 16; byteidx++) {
                uint8_t *out_lo = out + 64 + (size_t)byteidx * 16;
                uint8_t *out_hi = out + 64 + (size_t)(byteidx + 16) * 16;
                const uint8_t *col_src = blk_base + 2 + (size_t)byteidx;
                for (int r = 0; r < 32; r += 2) {
                    uint8_t b0 = col_src[(size_t)r * row_stride];
                    uint8_t b1 = col_src[(size_t)(r + 1) * row_stride];
                    out_lo[r >> 1] = (uint8_t)((b0 & 0x0Fu) | ((b1 & 0x0Fu) << 4));
                    out_hi[r >> 1] = (uint8_t)((b0 >> 4) | (b1 & 0xF0u));
                }
            }
        }
    }
    free(tmp);
    return 0;
}

/* ---------------- Q8_0 legacy -> 8x8 tiled ----------------
 * per 8-row group: d[8] f16 scales, then qs[k*32+m*4+i] = row m block b
 * elem (k*4+i). Output = 272 B per 32-col block.
 * [SHS 公理映射: blas_matrix_block_natural_isomorphism (block_matrix_assoc_natural)]
 * 连续 4B 子块同构整体搬移，替代内层逐字节散列拷贝。 */
int repack_q8_0_8x8_inplace(uint8_t *buf, int rows, int cols) {
    if (rows % 8 != 0 || cols % 32 != 0) return -1;
    int nb = cols / 32;
    size_t row_stride = (size_t)nb * 34;
    size_t group_bytes = (size_t)nb * 272;
    uint8_t *tmp = (uint8_t *)malloc(group_bytes);
    if (!tmp) return -1;
    for (int r0 = 0; r0 < rows; r0 += 8) {
        uint8_t *dst = buf + (size_t)(r0 >> 3) * group_bytes;
        memcpy(tmp, dst, group_bytes);
        for (int b = 0; b < nb; b++) {
            uint8_t *out = dst + (size_t)b * 272;
            const uint8_t *blk_base = tmp + (size_t)b * 34;
            for (int m = 0; m < 8; m++)
                memcpy(out + (size_t)m * 2, blk_base + (size_t)m * row_stride, 2);
            for (int k = 0; k < 8; k++)
                for (int m = 0; m < 8; m++)
                    memcpy(out + 16 + (size_t)k * 32 + (size_t)m * 4,
                           blk_base + (size_t)m * row_stride + 2 + (size_t)k * 4, 4);
        }
    }
    free(tmp);
    return 0;
}

/* ---------------- Q8_0 legacy -> 4x4 (block_q8_0x4) ----------------
 * per 4-row group: d[4] f16 scales, then qs[k*16+m*4+i]. 136 B/block.
 * [SHS 公理映射: blas_matrix_block_natural_isomorphism (block_matrix_assoc_natural)] */
int repack_q8_0_4x4_inplace(uint8_t *buf, int rows, int cols) {
    if (rows % 4 != 0 || cols % 32 != 0) return -1;
    int nb = cols / 32;
    size_t row_stride = (size_t)nb * 34;
    size_t group_bytes = (size_t)nb * 136;
    uint8_t *tmp = (uint8_t *)malloc(group_bytes);
    if (!tmp) return -1;
    for (int r0 = 0; r0 < rows; r0 += 4) {
        uint8_t *dst = buf + (size_t)(r0 >> 2) * group_bytes;
        memcpy(tmp, dst, group_bytes);
        for (int b = 0; b < nb; b++) {
            uint8_t *out = dst + (size_t)b * 136;
            const uint8_t *blk_base = tmp + (size_t)b * 34;
            for (int m = 0; m < 4; m++)
                memcpy(out + (size_t)m * 2, blk_base + (size_t)m * row_stride, 2);
            for (int k = 0; k < 8; k++)
                for (int m = 0; m < 4; m++)
                    memcpy(out + 8 + (size_t)k * 16 + (size_t)m * 4,
                           blk_base + (size_t)m * row_stride + 2 + (size_t)k * 4, 4);
        }
    }
    free(tmp);
    return 0;
}

/* dispatch: 唯一默认 8x8 tiled */
int repack_q8_0_tiled_inplace(uint8_t *buf, int rows, int cols) {
    return repack_q8_0_8x8_inplace(buf, rows, cols);
}

/* ---------------- Q4_0 legacy -> 8x8l (x8 decode copy, pure scalar) ----------------
 * 引擎行 3494-3520 同源拷贝（无 NEON 差异）。 */
int repack_q4_0_8x8l(uint8_t *__restrict dst, const uint8_t *__restrict src,
                     int rows, int cols) {
    if (rows % 8 != 0 || cols % 32 != 0) return -1;
    int nb = cols / 32;
    size_t src_row = (size_t)nb * 18;
    for (int r0 = 0; r0 < rows; r0 += 8) {
        for (int b = 0; b < nb; b++) {
            uint8_t *out = dst + (size_t)(r0 >> 3) * nb * 144 + (size_t)b * 144;
            const uint8_t *sb = src + (size_t)r0 * src_row + (size_t)b * 18;
            for (int r = 0; r < 8; r++)
                memcpy(out + (size_t)r * 2, sb + (size_t)r * src_row, 2);
            for (int i = 0; i < 16; i++) {
                uint64_t e;
                memcpy(&e, sb + (size_t)(i & 7) * src_row + 2 + (size_t)(i >> 3) * 8, 8);
                e ^= 0x8888888888888888ULL;
                memcpy(out + 16 + (size_t)i * 8, &e, 8);
            }
        }
    }
    return 0;
}

/* ================================================================
 * Conv int8 离线重排内核（VQF v4 阶段 P1）
 * [SHS 公理映射: blas_matrix_block_natural_isomorphism +
 *                tensor_5fcontraction_02096446]
 * 将运行期 yconv_cpu_dotprod (nw) 与 yime_pack_a (iw) 的惰性重排
 * 折叠到离线转换期，实现运行期 m->blob 零拷贝直挂。
 * ================================================================ */

/* CHW [OC, K] (K = C * KH * KW) -> HWC tap-major [OC, Kp] (Kp = (K+31)&~31)
 * 逐字节等价于 vllm_yolo.c 中 yconv_cpu_dotprod (hwc=1) 的 nw 构造。 */
int repack_conv_q8_hwc_tap(int8_t *__restrict dst, const int8_t *__restrict src,
                           int OC, int C, int KH, int KW) {
    if (OC <= 0 || C <= 0 || KH <= 0 || KW <= 0) return -1;
    int K = C * KH * KW;
    int Kp = (K + 31) & ~31;
    int KHW = KH * KW;
    memset(dst, 0, (size_t)OC * (size_t)Kp);
    for (int o = 0; o < OC; o++) {
        int8_t *wo = dst + (size_t)o * Kp;
        const int8_t *so = src + (size_t)o * K;
        for (int ic = 0; ic < C; ic++)
            for (int tap = 0; tap < KHW; tap++)
                wo[tap * C + ic] = so[ic * KHW + tap];
    }
    return 0;
}

/* HWC tap-major [OC, Kp] -> CHW [OC, K] 逆向无损还原（供逃生舱回退） */
int unpack_conv_q8_hwc_tap(int8_t *__restrict dst, const int8_t *__restrict src,
                           int OC, int C, int KH, int KW) {
    if (OC <= 0 || C <= 0 || KH <= 0 || KW <= 0) return -1;
    int K = C * KH * KW;
    int Kp = (K + 31) & ~31;
    int KHW = KH * KW;
    for (int o = 0; o < OC; o++) {
        int8_t *do_row = dst + (size_t)o * K;
        const int8_t *wo = src + (size_t)o * Kp;
        for (int ic = 0; ic < C; ic++)
            for (int tap = 0; tap < KHW; tap++)
                do_row[ic * KHW + tap] = wo[tap * C + ic];
    }
    return 0;
}

/* HWC tap-major [OC, Kp] -> SpacemiT K1 IME pack_a 布局 [mt][kb][mv][32]
 * 要求 OC % 8 == 0 且 Kp % 8 == 0（Kp 已对齐到 32，天然满足 Kp % 8 == 0）。
 * 逐字节等价于 vllm_yolo.c 中的 yime_pack_a(Wq, OC, Kp)。 */
int repack_conv_q8_ime_k1(int8_t *__restrict dst, const int8_t *__restrict src_hwc_kp,
                          int OC, int Kp) {
    if (OC <= 0 || (OC & 7) != 0 || Kp <= 0 || (Kp & 7) != 0) return -1;
    int mt_n = OC >> 3, kb_n = Kp >> 3;
    for (int mt = 0; mt < mt_n; mt++) {
        for (int kb = 0; kb < kb_n; kb++) {
            int8_t *d = dst + ((size_t)mt * kb_n + kb) * 64;
            for (int mv = 0; mv < 2; mv++)
                for (int q = 0; q < 4; q++)
                    memcpy(d + mv * 32 + q * 8,
                           src_hwc_kp + (size_t)(mt * 8 + mv * 4 + q) * Kp + kb * 8, 8);
        }
    }
    return 0;
}

/* SpacemiT K1 IME pack_a [mt][kb][mv][32] -> HWC tap-major [OC, Kp] 逆向无损还原 */
int unpack_conv_q8_ime_k1(int8_t *__restrict dst_hwc_kp, const int8_t *__restrict src_ime,
                          int OC, int Kp) {
    if (OC <= 0 || (OC & 7) != 0 || Kp <= 0 || (Kp & 7) != 0) return -1;
    int mt_n = OC >> 3, kb_n = Kp >> 3;
    for (int mt = 0; mt < mt_n; mt++) {
        for (int kb = 0; kb < kb_n; kb++) {
            const int8_t *s = src_ime + ((size_t)mt * kb_n + kb) * 64;
            for (int mv = 0; mv < 2; mv++)
                for (int q = 0; q < 4; q++)
                    memcpy(dst_hwc_kp + (size_t)(mt * 8 + mv * 4 + q) * Kp + kb * 8,
                           s + mv * 32 + q * 8, 8);
        }
    }
    return 0;
}
