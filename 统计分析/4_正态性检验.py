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
所有进行操作的列。请选择数值型数据，如果勾选了非数值类型数据，则会自动过滤，下个组件可能无法获取所有列。
"""

def execute(conn, inputs, params, outputs, reportFileName):
    #<editable>
	'''
	载入模块
	'''
	from scipy.stats import normaltest
	import db_utils
	import numpy as np
	import report_utils
	'''
	选择目标数据
	'''
	data_in = db_utils.query(conn, 'select '+ params['columns'] + ' from ' + inputs['data_in'])

	'''
	正态性检验
	'''
	data_in = data_in.select_dtypes(include=['number'])  # 筛选数值型数据
	p=normaltest(data_in, nan_policy=params['nan_policy'])[1]

	report.h1('正态性检验结果')
	report.h3('检验结果，当p<0.05时，可以证明数据不服从正态分布')
	p=pd.DataFrame(p)
	p.columns=['p值']

	report.table(np.around(p,decimals=4))

	'''
	生成报告
	'''
	report.writeToHtml(reportFileName)
    #</editable>
