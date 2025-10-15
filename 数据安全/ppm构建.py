import os

# 路径设置
header_path = r"C:\Users\zwq\Desktop\数据传输与安全\data\flower_ppm_header.txt"
raw_path = r"C:\Users\zwq\Desktop\数据传输与安全\data\flower-bsq-u8be-3x1512x2268.raw"
output_path = r"C:\Users\zwq\Desktop\数据传输与安全\data\flower_with_header.ppm"

# 读取 header 内容
with open(header_path, 'rb') as f_header:
    header_data = f_header.read()

# 读取 raw 数据
with open(raw_path, 'rb') as f_raw:
    raw_data = f_raw.read()

# 写入输出文件（先写 header，再写 raw 数据）
with open(output_path, 'wb') as f_out:
    f_out.write(header_data)
    f_out.write(raw_data)

print(f"✅ 已生成带文件头的 PPM 文件：{output_path}")
