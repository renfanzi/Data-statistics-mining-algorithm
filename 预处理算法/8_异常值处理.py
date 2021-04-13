# <editable>
# 在此处添加算法描述
# </editable>
# conn：             数据库连接
# inputs：           输入数据集合，数据类型：list， 存储组件输入节点对应的数据，
#                    通过输入节点的key获取数据，例如配置的key为“input1”, 那么inputs["input1"]
#                    即为该节点对应的数据表
# params：           参数集合，数据类型：list， 存储，获取的规则与inputs一致。需要注意的是：
#                    params中参数的值都是字符类型的，需要在代码中进行数据类型转换，比如：
#                    int(params["centers"])
# outputs：          存储规则参见inputs
# reportFileName：   算法运行报告文件的存储路径
# 返回值(可选)：     如果函数用于训练模型，则必须返回模型对象

"""
columns
特征
选择添加需要进行异常值处理的列
"""


def execute(conn, inputs, params, outputs, reportFileName):
    #<editable>
    import pyh
    import report_utils
    import db_utils
    import pandas as pd
    import warnings
    warnings.filterwarnings('ignore')
    '''
    选择目标数据
    '''
    data_in = db_utils.query(conn, 'select *' + ' from ' + inputs['data_in'])
    data_name = db_utils.query(conn, 'select '+ params['columns'] + ' from ' + inputs['data_in'])
    '''
    找出缺失值
    '''
    def outRange(Ser1):
        QL = Ser1.quantile(float(params['upper_quantile']))
        QU = Ser1.quantile(float(params['lower_quantile']))
        IQR = QU-QL
        Ser1.loc[Ser1>(QU+1.5*IQR)] = None
        Ser1.loc[Ser1<(QL-1.5*IQR)] = None
        return Ser1
    names = data_name.columns
    for j in names:
        data_in[j] = outRange(data_in[j])
    '''
    报告
    '''
    report.h1('异常值处理报告')
    report.h3('各列数据的异常值的数量')
    data_sum = pd.DataFrame(data_in.isnull().sum()).reset_index()
    report.table(data_sum)

    '''
    对异常值处理
    '''

    '''
    异常值处理方法：
        1. 删除异常值
        2. 中位数插补
        3. 众数插补
        4. 均值插补
        5. 线性插补
        6. 多项式插补
    '''
    if (params['method'] == 'drop'):
        data_out = data_in.dropna()
    elif (params['method'] == 'Median_interpolation'):
        data_in = data_in.select_dtypes(include=['number'])
        data_out = data_in.fillna(data_in.median())
    elif (params['method'] == 'Mode_interpolation'):
        data_in = data_in.select_dtypes(include=['number'])
        data_out = data_in.fillna(data_in.mode())
    elif (params['method'] == 'slinear'):
        data_in = data_in.select_dtypes(include=['number'])
        data_out = data_in.interpolate(method = 'slinear')
    elif (params['method'] == 'quadratic'):
        data_in = data_in.select_dtypes(include=['number'])
        data_out = data_in.interpolate(method = 'quadratic')
    else :
        data_in = data_in.select_dtypes(include=['number'])
        data_out = data_in.fillna(data_in.mean())
    data_out = pd.DataFrame(data_out)

    '''
    将结果写出
    '''
    report.writeToHtml(reportFileName)
    db_utils.dbWriteTable(conn, outputs['data_out'], data_out)

    #</editable>
