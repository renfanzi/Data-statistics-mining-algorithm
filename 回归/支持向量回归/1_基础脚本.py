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
"""

# 基础参数
"""
kernel
核函数
rbf
指定要在算法中使用的内核类型

degree
多项式阶数
2
正整数。该参数仅在内核为多项式核函数时起作用。

gamma
核系数
auto
正float类型或auto。该参数仅在内核为多项式、高斯或sigmoid核函数时起作用。auto表示没有传递明确的gamma值。

coef0
独立项
0.0
正float类型。该参数仅在内核为多项式或sigmoid核函数时起作用。

C
惩罚系数
1.0
正float类型。错误的惩罚系数。
"""

# 高级参数
"""
tol
容错率
1e-3
(0,1)之前的float类型。模型停止训练的容错标准。

epsilon
距离误差
0.1
float类型。训练集中的样本需满足模型拟合值与实际值的误差。

shrinking
收缩启发式
True
是否使用收缩启发式。

cache_size
缓存大小（MB）
200
float类型。当数据较大时，指定内核缓存的大小。（以MB为单位）。

max_iter
最大迭代次数
-1
int类型。进行迭代次数的上限。-1为无限制
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
    from sklearn.svm import SVR
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
    if params['gamma']=='auto' or params['gamma']=='scale':gamma=params['gamma']
    else:gamma=float(params['gamma'])
    model=SVR(kernel=params['kernel'], degree=int(params['degree']), gamma=gamma, coef0=float(params['coef0']), tol=float(params['tol']), C=float(params['C']), epsilon=float(params['epsilon']), shrinking='True'==params['shrinking'], cache_size=int(params['cache_size']), verbose=False, max_iter=int(params['max_iter']))

    '''
    模型训练与拟合
    '''
    model.fit(X_train, y_train.values.ravel())
    y_ov = model.predict(X_train)

    '''
    模型参数
    '''
    params_name=['核函数','多项式阶数','核系数','独立项','惩罚系数','容错率','距离误差','收缩启发式','缓存大小（MB）','最大迭代次数']
    params_value=[params['kernel'],params['degree'],params['gamma'],params['coef0'],params['C'],params['tol'],params['epsilon'],params['shrinking'],params['cache_size'],params['max_iter']]
    pars=pd.DataFrame([params_name,params_value]).T
    pars.columns=['参数名称','参数值']
    report.h3('模型参数')
    report.p("需要配置的参数及其取值如下。")
    report.table(pars)

    '''
    模型属性
    '''
    if params['kernel']=='linear':
        report.h3('模型属性')
        fol = params['label'] + '_pred = '
        intercept=model.intercept_[0]
        for i in range(len(model.coef_[0])):
            if i>0:
                if model.coef_[0][i]>0:fol += ' + '+str(model.coef_[0][i])+'*'+X_train.columns[i]
                else:fol +=' - '+str(abs(model.coef_[0][i]))+'*'+X_train.columns[i]
            elif model.coef_[0][0]>0:fol += str(intercept)+' + '+str(model.coef_[0][0])+'*'+X_train.columns[0]
            else:fol += str(intercept)+' - '+str(abs(model.coef_[0][0]))+'*'+X_train.columns[0]
        md_info1=pd.DataFrame({'模型公式':[fol]})
        report.table(md_info1)
    else:pass

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
    data_out=pd.concat([X_train,y_train,pd.DataFrame({'predict_value':y_ov})],axis=1)
    data_out['predict_value']=data_out.apply(lambda x:'%.2f'%x['predict_value'],axis=1)
    report.h3('模型拟合情况')
    report.p('取'+str(len(X_train))+'条数据作为训练集，建立支持向量回归模型，得到的预测值与真实值对比图如下图所示。')
    X_label=[str(i) for i in X_train.index]
    plt.figure(figsize=(6.0, 4.0))
    plt.style.use('ggplot')
    plt.plot(X_label, y_train, marker='*',label='origin')
    plt.plot(X_label, y_ov, marker='.',alpha=0.7,label='prediction')
    if len(X_train)>10 and (len(X_train)-1)%10<5:plt.xticks(np.linspace(0,np.ceil(len(X_train)/5)*5-1,5))
    elif len(X_train)>10 and (len(X_train)-1)%10>5:plt.xticks(np.linspace(0,np.ceil(len(X_train)/10)*10-1,10))
    else:plt.xticks(np.linspace(0,len(X_train)-1,len(X_train)))
    plt.legend()
    plt.title('Fitting of SVR Regression Model')
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
