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
col
数据
数据整体情况

sequence
特征
进行操作的列
"""


def execute(conn, inputs, params, outputs, reportFileName):
    #<editable>
	'''
	载入模块
	'''
	import warnings
	warnings.filterwarnings("ignore")
	from statsmodels.sandbox.stats.diagnostic import acorr_ljungbox
	import pyh
	import report_utils
	import db_utils
	import pandas as pd

	'''
	选择目标数据
	'''
	data_in = db_utils.query(conn, 'select '+ params['sequence'] + ' from ' + inputs['data_in'])


	'''
	纯随机性检验
	'''
	data_in = data_in.dropna()
	sequence = data_in[params['sequence']]
	p = acorr_ljungbox(sequence)[1]
	data_out = pd.DataFrame({'lags':range(1,len(p)+1), 'pvalue':p})

	report.p("如果p值小于0.05时,可以证明通过白噪声检验！")
	report.table(data_out)

	'''
	生成报告
	'''
	report.writeToHtml(reportFileName)
	'''
	将结果写出
	'''
	db_utils.dbWriteTable(conn, outputs['data_out'], data_out)
    #</editable>
