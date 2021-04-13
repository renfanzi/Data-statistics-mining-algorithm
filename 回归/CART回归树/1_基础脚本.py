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
模型训练的特征。请选择数值型数据

label
标签
模型训练的标签。请选择数值型数据

add_col
新列名
predict_value
新增的预测列的列名。
"""

# 基础参数
"""
criterion
切分评价准则
mse
切分时的评价准则

splitter
切分原则
best
切分原则

max_depth
树的最大深度
None
树的最大深度

min_samples_split
子树划分所需最小样本数
2
子树继续划分所需最小样本数

min_samples_leaf
叶子节点最少样本数
1
叶子节点最少样本数
"""

# 高级参数
"""
min_weight_fraction_leaf
叶子节点最小的样本权重和
0.
非负数类型。叶子节点最小的样本权重和

max_features
最大特征数
None
模型保留最大特征数。可输入int, float类型的数值，也可选择输入auto（原特征数）, sqrt（开方）, log2, None（原特征数）

max_leaf_nodes
最大叶子节点数
None
正float、int类型或None。最大叶子节点数

min_impurity_decrease
节点划分最小减少不纯度
0.
非负数类型。节点划分最小减少不纯度

presort
预排序
False
数据是否预排序
"""



def execute(conn, inputs, params, outputs, reportFileName):
    #<editable>
	'''
	载入模块
	'''
	import matplotlib.pyplot as plt
	import numpy as np
	import pandas as pd
	from sklearn.tree import DecisionTreeRegressor
	from sklearn.metrics import explained_variance_score,mean_squared_error,mean_absolute_error,r2_score
	from sklearn.externals.six import StringIO
	from sklearn import tree
	import pydot
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
	#获取参数
	def getparmas(x):
	    if x == 'None':return None
	    elif type(eval(x))==float:return float(x)
	    elif type(eval(x))==int:return int(x)
	    else:return x
	max_depth = getparmas(params['max_depth'])
	min_samples_split = getparmas(params['min_samples_split'])
	min_samples_leaf = getparmas(params['min_samples_leaf'])
	max_features = getparmas(params['max_features'])
	max_leaf_nodes  = getparmas(params['max_leaf_nodes'])
	model = DecisionTreeRegressor(criterion=params['criterion'], splitter=params['splitter'], max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=float(params['min_weight_fraction_leaf']), max_features=max_features, random_state=None, max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=float(params['min_impurity_decrease']),presort=True==params['presort'])

	'''
	模型训练与拟合
	'''
	model.fit(X_train,y_train)
	y_ov = model.predict(X_train)

	'''
	模型参数
	'''
	params_name=['切分评价准则','切分原则','树的最大深度','子树划分所需最小样本数','叶子节点最少样本数','叶子节点最小的样本权重和','最大特征数','最大叶子节点数','节点划分最小减少不纯度','预排序']
	params_value=[params['criterion'],params['splitter'],params['max_depth'],params['min_samples_split'],params['min_samples_leaf'],params['min_weight_fraction_leaf'],params['max_features'],params['max_leaf_nodes'],params['presort']]
	pars=pd.DataFrame([params_name,params_value]).T
	pars.columns=['参数名称','参数值']
	report.h3('模型参数')
	report.p("需要配置的参数及其取值如下。")
	report.table(pars)

	'''
	模型属性
	'''
	report.h3('模型属性')
	md_info1=pd.DataFrame({'模型属性':['输入特征数','输出特征数'],'属性值':[model.n_features_,model.max_features_]})
	report.table(md_info1.loc[:,['模型属性','属性值']])

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
	data_out=pd.concat([X_train,y_train,pd.DataFrame({params['add_col']:list(y_ov)})],axis=1)
	data_out[params['add_col']]=data_out.apply(lambda x:'%.2f'%x[params['add_col']],axis=1)
	report.h3('模型拟合情况')
	report.p('取'+str(len(X_train))+'条数据作为训练集，建立CART数回归模型，得到的预测值与真实值对比图如下图所示。')

	X_label=[str(i) for i in X_train.index]
	plt.figure(figsize=(6.0, 4.0))
	plt.style.use('ggplot')
	plt.plot(X_label, y_train, marker='*',label='origin')
	plt.plot(X_label, y_ov, marker='.',alpha=0.7,label='prediction')
	if len(X_train)>10 and (len(X_train)-1)%10<5:plt.xticks(np.linspace(0,np.ceil(len(X_train)/5)*5-1,5))
	elif len(X_train)>10 and (len(X_train)-1)%10>5:plt.xticks(np.linspace(0,np.ceil(len(X_train)/10)*10-1,10))
	else:plt.xticks(np.linspace(0,len(X_train)-1,len(X_train)))
	plt.legend()
	plt.title('Fitting of CART Tree Model')
	plt.xlabel('index')
	plt.ylabel(y_train.columns.values[0])
	plt.tight_layout()
	plt.savefig('overview.png')
	report.image('overview.png')

	'''
	CART决策树图
	'''
	dot_data = StringIO()
	tree.export_graphviz(model, out_file=dot_data)
	graph = pydot.graph_from_dot_data(dot_data.getvalue())
	graph[0].write_dot('model.dot')
	graph[0].write_png('model.png')
	report.h3('CART决策树图')
	report.image('model.png')

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
