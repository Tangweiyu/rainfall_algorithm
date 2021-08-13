# 封装api，接收Rainfall01,Rainfall03,Rainfall06,Rainfall24的输入，输出0，1值，其中0表示不发生事故，1表示发生事故

import pandas as pd
from flask import Flask, jsonify,request
import traceback
import joblib
import json

app = Flask(__name__)
@app.route('/ ', methods=['GET']) #在URL里使用/来进行调用api
def predict():
    if model:
        try:
            # 默认返回内容
            return_dict = {'return_code': '200', 'return_info': '处理成功', 'result': None}
            # 判断传入的json数据是否为空
            if len(request.args) == 0:
                return_dict['return_code'] = '5004'
                return_dict['return_info'] = '请求参数为空'
                return json.dumps(return_dict, ensure_ascii=False)
            # 获取传入的params参数
            get_data = request.args.to_dict()
            Rainfall01 = get_data.get('Rainfall01')
            Rainfall03 = get_data.get('Rainfall03')
            Rainfall06 = get_data.get('Rainfall06')
            Rainfall24 = get_data.get('Rainfall24')
            # 填充为空的参数,第一个实时数据若为空则用0值填充，其他实时数据填充参考处理后的第一个实时数据
            temp = 0
            if Rainfall01 == None or Rainfall01 == "":
                Rainfall01 = 0
                temp = Rainfall01
            else:
                temp = Rainfall01
            if Rainfall03 == None or Rainfall03 == "":
                Rainfall03 = temp
            else:
                temp = Rainfall03
            if Rainfall06 == None or Rainfall06 == "":
                Rainfall06 = temp
            else:
                temp = Rainfall06
            if Rainfall24 == None or Rainfall24 == "":
                Rainfall24 = temp
            # 下面是处理输入数据，转换为模型接收数据
            Rainfall01 = float(Rainfall01)
            Rainfall03 = float(Rainfall03)
            Rainfall06 = float(Rainfall06)
            Rainfall24 = float(Rainfall24)
            li = []
            li.append(Rainfall01)
            li.append(Rainfall03)
            li.append(Rainfall06)
            li.append(Rainfall24)
            query = pd.DataFrame([li],columns=['Rainfall01', 'Rainfall03', 'Rainfall06', 'Rainfall24'],dtype="float32")
            prediction = list(model.predict(query))[0]#预测结果
            return {'预测结果': str(prediction)}
        except:
            # 处理错误
            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')

if __name__ == '__main__':
    # 设置端口号
    port = 12345
    # 加载模型
    model = joblib.load("model.pkl")
    print('Model loaded')
    # 加载输入模型的属性列表
    model_columns = joblib.load("model_columns.pkl")
    print('Model columns loaded')
    # 服务部署运行
    app.run(port=port, debug=True)

