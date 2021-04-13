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
features
特征
模型训练的特征。

label
标签
模型训练的标签。

add_col
新列名
predict_value
新增的预测列的列名。
"""

# 基础参数
"""
alpha
L1项系数
1.0
一个常数用以与L1项相乘，等于0则为最小二乘。

fit_intercept
拟合截距
True
是否计算此模型的截距。

max_iter
最大迭代次数
500
进行迭代次数的上限
"""

# 高级参数
"""
normalize
归一化
False
特征量X是否在回归之前进行数据归一化。

precompute
预计算
False
是否使用预先计算的Gram矩阵来加速计算。

tol
容错率
1e-06
(0,1)之前的float类型。模型停止训练的容错标准。

warm_start
热启动
False
设置为True时，重用上一次调用的解决方案以适合初始化，否则，擦除以前的解决方案。

positive
强制正相关
False
是否强制系数为正。

selection
选择器
cyclic	
如果设置为“随机”，则每次迭代都会随机更新系数，而不是默认情况下按顺序循环使用要素。
"""

def execute(conn, inputs, params, outputs, reportFileName):
    #<editable>
	'''
	载入模块
	'''
	import matplotlib.pyplot as plt
	import numpy as np
	import pandas as pd
	from sklearn.externals import joblib
	from sklearn.metrics import explained_variance_score,mean_squared_error,mean_absolute_error,r2_score
	from sklearn.linear_model import Lasso
	import pyh
	import report_utils
	import db_utils

	'''
	载入数据
	'''
	X_train = db_utils.query(conn, 'select '+ params['features'] + ' from ' + inputs['data_in'])
	y_train = db_utils.query(conn, 'select '+ params['label'] + ' from ' + inputs['data_in'])


	'''
	建立模型
	'''
	import warnings
	warnings.filterwarnings("ignore")
	model = Lasso(alpha=float(params['alpha']),fit_intercept='True'==params['fit_intercept'], normalize='True'==params['normalize'],max_iter=int(params['max_iter']),precompute='True'==params['precompute'],tol=float(params['tol']),warm_start='True'==params['warm_start'],positive='True'==params['positive'],selection=params['selection'])


	'''
	模型训练与拟合
	'''
	model.fit(X_train, y_train)
	y_ov = model.predict(X_train)

	'''
	模型参数
	'''
	params_name=['L1项系数','拟合截距','最大迭代次数','归一化','预计算','容错率','热启动','强制正相关','选择器']
	params_value=[params['alpha'],params['fit_intercept'],params['max_iter'],params['normalize'],params['precompute'],params['tol'],params['warm_start'],params['positive'],params['selection']]
	pars=pd.DataFrame([params_name,params_value]).T
	pars.columns=['参数名称','参数值']
	report.h3('模型参数')
	report.p("需要配置的参数及其取值如下。")
	report.table(pars)


	'''
	模型属性
	'''
	report.h3('模型属性')
	fol = params['label'] + '_pred = '
	intercept=(model.intercept_[0] if 'True'==params['fit_intercept'] else 0)
	for i in range(len(model.coef_)):
	    if i>0:
	        if model.coef_[i]>0:fol += ' + '+str(model.coef_[i])+'*'+X_train.columns[i]
	        else:fol +=' - '+str(abs(model.coef_[i]))+'*'+X_train.columns[i]
	    elif model.coef_[0]>0:fol += str(intercept)+' + '+str(model.coef_[0])+'*'+X_train.columns[0]
	    else:fol += str(intercept)+' - '+str(abs(model.coef_[0]))+'*'+X_train.columns[0]
	md_info1=pd.DataFrame({'模型公式':[fol]})
	report.table(md_info1)


	'''
	模型指标
	'''
	report.h3('模型评价指标')
	report.p('模型拟合效果指标如下。')
	md_info=pd.DataFrame()
	md_info['重要指标']=['MSE','RMSE','MAE','EVS','R-Squared']
	md_info['值']=['%.2f'%mean_squared_error(y_train, y_ov),'%.2f'%np.sqrt(mean_squared_error(y_train, y_ov)),'%.2f'%mean_absolute_error(y_train,y_ov),'%.2f'%explained_variance_score(y_train, y_ov),'%.2f'%r2_score(y_train, y_ov)]
	report.table(md_info)


	'''
	模型拟合情况
	'''
	data_out=pd.concat([X_train,y_train,pd.DataFrame({params['add_col']:y_ov})],axis=1)
	data_out[params['add_col']]=data_out.apply(lambda x:'%.2f'%x[params['add_col']],axis=1)
	report.h3('模型拟合情况')
	report.p('取'+str(len(X_train))+'条数据作为训练集，建立LASSO回归模型，得到的预测值与真实值对比图如下图所示。')
	X_label=[str(i) for i in X_train.index]
	plt.figure(figsize=(6.0, 4.0))
	plt.style.use('ggplot')
	plt.plot(X_label, y_train, marker='*',label='origin')
	plt.plot(X_label, y_ov, marker='.',alpha=0.7,label='prediction')
	if len(X_train)>10 and (len(X_train)-1)%10<5:plt.xticks(np.linspace(0,np.ceil(len(X_train)/5)*5-1,5))
	elif len(X_train)>10 and (len(X_train)-1)%10>5:plt.xticks(np.linspace(0,np.ceil(len(X_train)/10)*10-1,10))
	else:plt.xticks(np.linspace(0,len(X_train)-1,len(X_train)))
	plt.legend()
	plt.title('Fitting of LASSO Regression Model')
	plt.xlabel('index')
	plt.ylabel(y_train.columns.values[0])
	plt.tight_layout()
	plt.savefig('overview.png')
	report.image('overview.png')

	# 保存模型
	modelFile = 'model.pkl'
	joblib.dump(model, modelFile)


	'''
	生成报告
	'''
	report.writeToHtml(reportFileName)

	'''
	将结果写出
	'''
	db_utils.dbWriteTable(conn, outputs['data_out'], data_out)

	return(model)
    #</editable>
