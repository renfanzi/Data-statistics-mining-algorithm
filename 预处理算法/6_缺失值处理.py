# <editable>
# 在此处添加算法描述
# </editable>
# conn：             数据库连接
# inputs：           输入数据集合，数据类型：list， 存储组件输入节点对应的数据，
#                    通过输入节点的key获取数据，例如配置的key为“input1”, 那么inputs$input1
#                    即为该节点对应的数据表
# params：           参数集合，数据类型：list， 存储，获取的规则与inputs一致。需要注意的是：
#                    params中参数的值都是字符类型的，需要在代码中进行数据类型转换，比如：
#                    as.integer(params$centers)
# outputs：          存储规则参见inputs
# reportFileName：   算法运行报告文件的存储路径
# 返回值(可选)：     如果函数用于训练模型，则必须返回模型对象

"""
columns	
特征
必选。选用中位数插值法、众数插值法、均值插值法时，请选择数值型数据，如果勾选了非数值类型数据，则会自动过滤，下个组件可能无法获取所有列。勾选的列将传入下一个组件。
"""



def execute(conn, inputs, params, outputs, reportFileName):
    #<editable>
    '''
    载入模块
    '''
    import db_utils
    import pandas as pd
    '''
    选择目标数据
    '''
    data_in = db_utils.query(conn, 'select '+ params['columns'] + ' from ' + inputs['data_in'])

    '''
    缺失值处理
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
    db_utils.dbWriteTable(conn, outputs['data_out'], data_out)
    #</editable>
