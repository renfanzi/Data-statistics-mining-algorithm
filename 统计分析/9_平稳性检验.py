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
时序特征	
进行操作的时序列。如果该列数据含有缺失值，则会自动删除。
"""


def execute(conn, inputs, params, outputs, reportFileName):
    #<editable>
	'''
	载入模块
	'''
	import warnings
	from statsmodels.graphics.tsaplots import plot_acf  # 绘制自相关图
	from statsmodels.tsa.stattools import adfuller as ADF  # 单位根检验
	import matplotlib.pyplot as plt
	import pyh
	import report_utils
	import db_utils
	warnings.filterwarnings("ignore")

	'''
	选择目标数据
	'''
	data_in = db_utils.query(conn, 'select '+ params['sequence'] + ' from ' + inputs['data_in'])
	data_in = data_in.dropna()
	'''
	平稳性检验
	'''
	sequence = data_in[params['sequence']]
	adf_result = ADF(sequence)
	test_statistic = adf_result[0]
	p_value = adf_result[1]
	use_lag = adf_result[2]
	nobs = adf_result[3]
	critical_1 = adf_result[4]['5%']
	critical_5 = adf_result[4]['1%']
	critical_10 = adf_result[4]['10%']
	report.h1('平稳性检验结果')
	report.h3('检验结果')
	report.p('Test statistic：' + str(test_statistic))
	report.p(' p-value：' + str(p_value))
	report.p('Number of lags used：' + str(use_lag))
	report.p('Number of observations used for the ADF regression and calculation of the critical values：' + str(nobs))
	report.p('Critical values for the test statistic at the 5 %：' + str(critical_1))
	report.p('Critical values for the test statistic at the 1 %：' + str(critical_5))
	report.p('Critical values for the test statistic at the 10 %：' + str(critical_10))

	'''
	自相关图
	'''
	fig = plt.figure(figsize=(10,4))
	ax1 = fig.add_subplot(111)
	plot_acf(sequence, ax = ax1,fft = True)
	plt.savefig('acf.png')
	report.h3('ACF')
	report.image('acf.png')

	'''
	生成报告
	'''
	report.writeToHtml(reportFileName)
    #</editable>
