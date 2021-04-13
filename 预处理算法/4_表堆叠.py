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
columns_left
表一特征
第一个进行合并的表的列

columns_right
表二特征
第二个进行合并的列

method
合并方法
1       （默认值）
可选择行合并，列合并;需要注意的是，选择列合并时，两个表的行数需要一样;选择行合并时，两个表的列数需要一样。
"""


def execute(conn, inputs, params, outputs, reportFileName):
    #<editable>

	'''
	载入模块
	'''
	import pandas as pd
	import db_utils
	'''
	选择目标数据
	'''
	if params['columns_left']=='':
	    left = db_utils.query(conn, 'select * from ' + inputs['left'])
	else:
	    left = db_utils.query(conn, 'select '+ params['columns_left'] + ' from ' + inputs['left'])
	if params['columns_right']=='':
	    right = db_utils.query(conn, 'select * from ' + inputs['right'])
	else:
	    right = db_utils.query(conn, 'select '+ params['columns_right'] + ' from ' + inputs['right'])
	'''
	合并数据
	'''
	data_out = pd.concat([left, right], axis = int(params['method']))
	'''
	将结果写出
	'''
	db_utils.dbWriteTable(conn, outputs['data_out'], data_out)

    #</editable>
