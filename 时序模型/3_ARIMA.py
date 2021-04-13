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


# 字段属性
"""
timeseries
时序列
time
时间列
"""

# 基础参数
"""
periods
预测周期数
10
模型往后预测的周期数。大于0的整数

p
自回归项数p
0
AR模型系数。0和正整数

d
差分次数d
0
非平稳数据处理为平稳数据的差分次数

q
移动平均项数q
0	
MA模型系数。0和正整数
"""



def execute(conn, inputs, params, outputs, reportFileName):
    #<editable>
	'''
	载入模块
	'''
	import warnings
	from statsmodels.tsa.statespace.sarimax import SARIMAX
	from sklearn.metrics import explained_variance_score,mean_squared_error,mean_absolute_error,r2_score
	import numpy as np
	import pandas as pd
	import matplotlib.pyplot as plt
	import pyh
	import report_utils
	import db_utils

	'''
	选择目标数据
	'''
	timeseries = db_utils.query(conn, 'select '+ params['timeseries'] + ' from ' + inputs['data_in'])
	time = db_utils.query(conn, 'select '+ params['time'] + ' from ' + inputs['data_in'])
	time[params['time']]=pd.to_datetime(time[params['time']])
	order = (int(params['p']), int(params['d']), int(params['q']))

	'''
	建立、训练和拟合模型
	'''
	warnings.filterwarnings("ignore")
	model=SARIMAX(timeseries,order=order).fit(disp=0)
	ts_pred=model.predict(1,len(timeseries)+int(params['periods']),typ='levels').reset_index(drop=True)

	'''
	模型参数
	'''
	pars=pd.DataFrame([['自回归项数p','差分次数d','移动平均项数q'],[int(params['p']),int(params['d']),int(params['q'])]]).T
	pars.columns=['参数名称','参数值']
	report.h3('模型参数')
	report.p("需要配置的参数及其取值如下。")
	report.table(pars)

	'''
	模型具体信息
	'''
	report.h3('模型具体信息')
	report.p('模型具体信息如下。')
	report.pre(model.summary())

	'''
	模型检验图
	'''
	plt.figure(figsize=(6.0, 4.0))
	model.plot_diagnostics()
	plt.tight_layout()
	plt.savefig('diagnostics.png')
	report.h3('模型检验图')
	report.p('模型各项检验信息如下图。')
	report.image('diagnostics.png')

	'''
	模型指标
	'''
	report.h3('模型评价指标')
	report.p('模型拟合效果指标如下。')
	md_info=pd.DataFrame()
	md_info['重要指标']=['MSE','RMSE','MAE','EVS','R-Squared']
	md_info['值']=['%.2f'%mean_squared_error(timeseries, ts_pred[:len(time)]),'%.2f'%np.sqrt(mean_squared_error(timeseries, ts_pred[:len(time)])),'%.2f'%mean_absolute_error(timeseries,ts_pred[:len(time)]),'%.2f'%explained_variance_score(timeseries, ts_pred[:len(time)]),'%.2f'%r2_score(timeseries, ts_pred[:len(time)])]
	report.table(md_info)

	'''
	模型拟合情况
	'''
	fit_data=pd.concat([timeseries,pd.DataFrame({timeseries.columns.values[0]+'_fitValue':ts_pred[:len(time)]})],axis=1)
	fit_data[timeseries.columns.values[0]+'_fitValue']=fit_data.apply(lambda x:'%.2f'%x[timeseries.columns.values[0]+'_fitValue'],axis=1)
	report.h3('模型拟合情况')
	report.p('建立ARIMA模型,设置自回归项数p、差分次数d和移动平均项数分别为'+str(int(params['p']))+'、'+str(int(params['d']))+'和'+str(int(params['q']))+'，得到的预测值与实际值对比图如下图所示。')
	plt.figure(figsize=(6.0, 4.0))
	plt.style.use('ggplot')
	plt.plot(time, timeseries, marker='*', label='origin')
	plt.plot(time, ts_pred[:len(time)], marker='.', alpha=0.7,label='fit')
	plt.legend()
	plt.title('Fitting of ARIMA Model')
	plt.xlabel('date')
	plt.ylabel(timeseries.columns.values[0])
	plt.tight_layout()
	plt.savefig('overview.png')
	report.image('overview.png')

	'''
	模型预测情况
	'''
	pre_index=pd.date_range(start=time[params['time']][len(time)-1],periods=int(params['periods']),freq=time[params['time']][len(time)-1]-time[params['time']][len(time)-2])
	report.h3('模型预测情况')
	report.p('设置预测周期数为'+str(int(params['periods']))+'，得到的预测值如下图所示。')
	pre_data=pd.DataFrame({timeseries.columns.values[0]+'_preValue':ts_pred[len(time):]})
	pre_data[timeseries.columns.values[0]+'_preValue']=pre_data.apply(lambda x:'%.2f'%x[timeseries.columns.values[0]+'_preValue'],axis=1)
	pre_data.index=pre_index
	plt.figure(figsize=(6.0, 4.0))
	plt.style.use('ggplot')
	plt.title('Prediction of ARIMA Model')
	plt.plot(time,timeseries, '.-', label='origin')
	plt.plot(pre_index,ts_pred[len(time):], '.-', label='prediction')
	plt.legend()
	plt.xlabel('date')
	plt.ylabel(timeseries.columns.values[0])
	plt.tight_layout()
	plt.savefig('pre_view.png')
	report.image('pre_view.png')
	fit_data=time
	pre_data=timeseries

	'''
	生成报告
	'''
	report.writeToHtml(reportFileName)

	'''
	将结果写出
	'''
	db_utils.dbWriteTable(conn, outputs['fit_data'], fit_data)
	db_utils.dbWriteTable(conn, outputs['pre_data'], pre_data)
    #</editable>
