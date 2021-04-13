# <editable>
# 在此处添加算法描述
# </editable>
# conn：             数据库连接
# model：            评估的模型
# inputs:            输入集合
# params:            参数
# outputs:           输出集合
# reportFileName：   评估报告文件的存储路径








def evaluate(conn, model, inputs, params, outputs, reportFileName):
    #<editable>
	'''
	载入模块
	'''
	import matplotlib.pyplot as plt
	import numpy as np
	import pandas as pd
	import itertools
	from sklearn.metrics import explained_variance_score,mean_squared_error,mean_absolute_error,r2_score
	import pyh
	import report_utils
	import db_utils

	'''
	载入数据
	'''
	X_test = db_utils.query(conn, 'select '+ params['features'] + ' from ' + inputs['data_in'])
	y_test = db_utils.query(conn, 'select '+ params['label'] + ' from ' + inputs['data_in'])

	'''
	模型预测
	'''
	y_pred = model.predict(X_test)

	'''
	模型评价
	'''
	report.h3('模型评价指标')
	report.p('取'+str(len(X_test))+'条数据作为测试集，检验模型效果，模型预测效果指标如下。')
	md_info2=pd.DataFrame()
	md_info2['重要指标']=['MSE','RMSE','MAE','EVS','R_Squared']
	md_info2['值']=['%.2f'%mean_squared_error(y_test, y_pred),'%.2f'%np.sqrt(mean_squared_error(y_test, y_pred)),'%.2f'%mean_absolute_error(y_test,y_pred),'%.2f'%explained_variance_score(y_test, y_pred),'%.2f'%r2_score(y_test, y_pred)]
	report.table(md_info2)
	report.h3('模型预测情况')
	report.p('得到的预测值与真实值对比图如下图所示。')

	X_label=[str(i) for i in X_test.index]
	plt.figure(figsize=(6.0, 4.0))
	plt.style.use('ggplot')
	plt.title('Prediction of Nearest Neighbors Regression models')
	plt.plot(X_label, y_test, marker='*', label='origin')
	plt.plot(X_label, y_pred, marker='.', alpha=0.7,label='prediction')
	if len(X_test)>10 and (len(X_test)-1)%10<5:plt.xticks(np.linspace(0,np.ceil(len(X_test)/5)*5-1,5))
	elif len(X_test)>10 and (len(X_test)-1)%10>5:plt.xticks(np.linspace(0,np.ceil(len(X_test)/10)*10-1,10))
	else:plt.xticks(np.linspace(0,len(X_test)-1,len(X_test)))
	plt.legend()
	plt.xlabel('index')
	plt.ylabel(params['label'])
	plt.tight_layout()
	plt.savefig('pre_view.png')
	report.image('pre_view.png')
	data_out=pd.concat([X_test,y_test,pd.DataFrame({params['add_col']:list(itertools.chain.from_iterable(y_pred))})],axis=1)
	data_out[params['add_col']]=data_out.apply(lambda x:'%.2f'%x[params['add_col']],axis=1)


	'''
	生成报告
	'''
	report.writeToHtml(reportFileName)

	'''
	将结果写出
	'''
	db_utils.dbWriteTable(conn, outputs['data_out'], data_out)

    #</editable>
