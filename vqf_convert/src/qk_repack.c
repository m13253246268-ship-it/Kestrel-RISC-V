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
 * 与 lane(2) 同源（legacy 18B/块），仅重排；体积不变（576 B/块）。 */
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
            /* scale：按行线性（r*2） */
            for (int r = 0; r < 32; r++)
                memcpy(out + (size_t)r * 2,
                       tmp + (size_t)r * row_stride + (size_t)b * 18, 2);
            /* data：列主序，偶行低 / 奇行高 */
            memset(out + 64, 0, 512);
            for (int c = 0; c < 32; c++) {
                int byteidx = c & 15;
                int hi = (c >> 4) & 1;
                for (int r = 0; r < 32; r++) {
                    uint8_t byte = tmp[(size_t)r * row_stride
                                     + (size_t)b * 18 + 2 + (size_t)byteidx];
                    uint8_t nib = hi ? (uint8_t)(byte >> 4) : (uint8_t)(byte & 0x0F);
                    uint8_t *p = out + 64 + (size_t)c * 16 + (size_t)(r >> 1);
                    if (r & 1) *p |= (uint8_t)(nib << 4);
                    else       *p |= (uint8_t)(nib & 0x0F);
                }
            }
        }
    }
    free(tmp);
    return 0;
}

/* ---------------- Q8_0 legacy -> 8x8 tiled ----------------
 * per 8-row group: d[8] f16 scales, then qs[k*32+m*4+i] = row m block b
 * elem (k*4+i). Output = 272 B per 32-col block. */
int repack_q8_0_8x8_inplace(uint8_t *buf, int rows, int cols) {
    if (rows % 8 != 0 || cols % 32 != 0) return -1;
    int nb = cols / 32;
    size_t group_bytes = (size_t)nb * 272;
    uint8_t *tmp = (uint8_t *)malloc(group_bytes);
    if (!tmp) return -1;
    for (int r0 = 0; r0 < rows; r0 += 8) {
        uint8_t *dst = buf + (size_t)(r0 >> 3) * group_bytes;
        memcpy(tmp, dst, group_bytes);
        for (int b = 0; b < nb; b++) {
            uint8_t *out = dst + (size_t)b * 272;
            for (int m = 0; m < 8; m++)
                memcpy(out + (size_t)m * 2,
                       tmp + (size_t)m * nb * 34 + (size_t)b * 34, 2);
            for (int k = 0; k < 8; k++)
                for (int m = 0; m < 8; m++)
                    for (int i = 0; i < 4; i++)
                        out[16 + k * 32 + m * 4 + i] =
                            tmp[(size_t)m * nb * 34 + (size_t)b * 34 + 2 + k * 4 + i];
        }
    }
    free(tmp);
    return 0;
}

/* ---------------- Q8_0 legacy -> 4x4 (block_q8_0x4) ----------------
 * per 4-row group: d[4] f16 scales, then qs[k*16+m*4+i]. 136 B/block. */
int repack_q8_0_4x4_inplace(uint8_t *buf, int rows, int cols) {
    if (rows % 4 != 0 || cols % 32 != 0) return -1;
    int nb = cols / 32;
    size_t group_bytes = (size_t)nb * 136;
    uint8_t *tmp = (uint8_t *)malloc(group_bytes);
    if (!tmp) return -1;
    for (int r0 = 0; r0 < rows; r0 += 4) {
        uint8_t *dst = buf + (size_t)(r0 >> 2) * group_bytes;
        memcpy(tmp, dst, group_bytes);
        for (int b = 0; b < nb; b++) {
            uint8_t *out = dst + (size_t)b * 136;
            for (int m = 0; m < 4; m++)
                memcpy(out + (size_t)m * 2,
                       tmp + (size_t)m * nb * 34 + (size_t)b * 34, 2);
            for (int k = 0; k < 8; k++)
                for (int m = 0; m < 4; m++)
                    for (int i = 0; i < 4; i++)
                        out[8 + k * 16 + m * 4 + i] =
                            tmp[(size_t)m * nb * 34 + (size_t)b * 34 + 2 + k * 4 + i];
        }
    }
    free(tmp);
    return 0;
}

/* dispatch: 默认 8x8（VLLM_Q8_8X8=0 回退 4x4），语义同引擎 */
static int g_q8_8x8 = -1;
static int q8_8x8_enabled(void) {
    if (g_q8_8x8 < 0) {
        const char *e = getenv("VLLM_Q8_8X8");
        g_q8_8x8 = (e && e[0] == '0') ? 0 : 1;
    }
    return g_q8_8x8;
}
int repack_q8_0_tiled_inplace(uint8_t *buf, int rows, int cols) {
    if (q8_8x8_enabled()) return repack_q8_0_8x8_inplace(buf, rows, cols);
    return repack_q8_0_4x4_inplace(buf, rows, cols);
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
