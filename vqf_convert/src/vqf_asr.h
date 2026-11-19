/* ================================================================
 * vqf_asr.h - Audio8-ASR ONNX bundle -> VQF v6 转换入口
 *
 * 与 tools/asr/onnx_to_vqf.py 语义等价（F32/BLOB、layout=0、目录名
 * t%06d + asr_metadata、头部 16..32 写 "ASR-NUMPY-v1"、尾部 FNV-1a
 * 覆盖「目录 + 数据区」）。产物字节级对拍 Python 参考实现。
 * ================================================================ */
#ifndef VQF_ASR_H
#define VQF_ASR_H

/* dir 内存在 metadata.json 且 format == audio8_asr_onnx_bundle 时返回 1 */
int vqf_asr_is_bundle(const char *dir);

/* 0 ok；非 0 失败（已打印原因） */
int vqf_asr_convert(const char *dir, const char *out, int w8);

#endif /* VQF_ASR_H */
