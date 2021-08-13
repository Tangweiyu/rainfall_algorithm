import requests, json
url='http://127.0.0.1:12345/'

# 所需要输入的数据格式：字符串类型，一共4个属性，每个属性输入数值的字符串类型即可


data="http://127.0.0.1:12345/?" + "Rainfall01=" + "1" + "&Rainfall03=" + "2" + "&Rainfall06=" + "3" + "&Rainfall24=" + "4"

result = requests.get(url, data).json()
print(result)