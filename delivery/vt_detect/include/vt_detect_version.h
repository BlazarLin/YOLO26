// 创建时间：2026-09-29；功能：统一 DLL 运行时与文件版本；目的：避免导出版本和 Windows 文件属性不一致。
#pragma once

#define VT_DETECT_VERSION_MAJOR 1
#define VT_DETECT_VERSION_MINOR 0
#define VT_DETECT_VERSION_PATCH 0
#define VT_DETECT_VERSION_TEXT "1.0.0"
#define VT_DETECT_FILE_VERSION 1,0,0,0
#define VT_DETECT_FILE_VERSION_TEXT "1.0.0.0"
