# 客户端调用服务器部署的api

import requests

# 请求头
header = {'Content-Type': 'application/json',
                  'Accept': 'application/json'}
# 所需要输入的数据格式：字符串类型，一共4个属性，每个属性输入数值的字符串类型即可

Rainfall01 = "1"
Rainfall03 = "2"
Rainfall06 = "3"
Rainfall024 = "4"
input_data = "http://127.0.0.1:12345/?" + "Rainfall01=" + Rainfall01 + "&Rainfall03=" + Rainfall03 + "&Rainfall06=" + Rainfall06 + "&Rainfall24=" + Rainfall024
print(input_data)
# 向服务器发送请求，请求的网址是http://127.0.0.1:12345 + 前面app.py里设置的“/”路径
resp = requests.get(input_data, headers= header)
# 返回的结果,eg：0
# 结果是字符串类型，里面只包含一条结果数据，代表当前输入数据的预测结果，返回0代表不发生事故，返回1代表发生事故
result = resp.json()
print(result)

