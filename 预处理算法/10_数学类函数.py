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
数据
必须包含待运算的列，勾选的列将传入下一个组件。

label
特征
进行操作的列
"""



def execute(conn, inputs, params, outputs, reportFileName):
    #<editable>
	'''
	导入模块
	'''
	import math
	import db_utils

	'''
	选择目标数据
	'''
	data_in = db_utils.query(conn, 'select '+ params['columns'] + ' from ' + inputs['data_in'])
	'''
	使用数学类函数
	'''
	fun = eval(params['method'])
	if data_in[params['label']].dtypes == 'float64' or data_in[params['label']].dtypes == 'int':
	    data_in[params['label']] = data_in[params['label']].apply(lambda x: fun(x))
	    data_out = data_in
	else:
	    raise ValueError('请选择数值型数据！')

	'''
	将结果写出
	'''
	db_utils.dbWriteTable(conn, outputs['data_out'], data_out)
    #</editable>
