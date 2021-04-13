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
时间数据，能够被pandas.to_datetime转化的字符类型数据或datetime类型数据

time
时间列
时序数据，数值型数据
"""
# 基础参数
"""
periods
预测周期数
10	
模型往后预测的周期数。大于0的整数
"""






def execute(conn, inputs, params, outputs, reportFileName):
    #<editable>
    '''
    载入模块
    '''
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.metrics import explained_variance_score,mean_squared_error,mean_absolute_error,r2_score
    import pyh
    import report_utils
    import db_utils

    '''
    载入数据
    '''
    time = db_utils.query(conn, 'select '+ params['time'] + ' from ' + inputs['data_in'])
    timeseries = db_utils.query(conn, 'select '+ params['timeseries'] + ' from ' + inputs['data_in'])
    if type(time[params['time']][0])==pd._libs.tslibs.timestamps.Timestamp:pass
    else:time[params['time']]=pd.to_datetime(time[params['time']])
    n=int(params['periods'])

    '''
    自定义函数
    '''
    def ident(timeseries):    #GM(1,1)模型系数求解
        B = np.array([[1]*2]*(len(timeseries)-1))
        fit_cum=np.cumsum(timeseries[params['timeseries']].values)
        for i in range(len(timeseries)-1):
            B[i][0] = (fit_cum[0]+fit_cum[i+1])*(-1.0)/ 2
        cf_B = np.dot(np.linalg.inv(np.dot(B.T,B)),B.T)
        tm_coef = np.dot(cf_B,timeseries[params['timeseries']].values[1:].T).T
        return fit_cum,tm_coef

    #GM(1,1)模型关系式
    def gm_pre(timeseries,cum_dt,tm_coef,n):
        pre_vl=tm_coef[1]/tm_coef[0] + (timeseries.values[0]-tm_coef[1]/tm_coef[0])*np.exp(tm_coef[0]*(n-1)*(-1))
        return pre_vl

    def gmm(timeseries,n):          #GM(1,1)模型预测值还原
        fit_cum,tm_coef=ident(timeseries)
        model_dt = np.ones(len(timeseries)+n)
        real_dt = np.ones(len(timeseries)+n)
        for i in range(len(model_dt)):
            real_dt[i] = gm_pre(timeseries,fit_cum,tm_coef,i)-gm_pre(timeseries,fit_cum,tm_coef,i-1)
        real_dt[0]=timeseries[params['timeseries']][0]
        return tm_coef,real_dt

    tm_coef,ts_pred=gmm(timeseries,n)

    '''
    模型参数
    '''
    pars=pd.DataFrame([['预测周期数',n]],columns=['参数名称','参数值'])
    report.h3('模型参数')
    report.p("需要配置的参数及其取值如下。")
    report.table(pars)

    '''
    模型信息
    '''
    report.h3('模型信息')
    if tm_coef[1]/tm_coef[0] <0:signal=' - '
    else:signal=' + '
    fol='X(k) = '+str(timeseries[params['timeseries']][0]-tm_coef[1]/tm_coef[0])+'*e^('+str(-tm_coef[0])+'*(k-1))'+signal+str(abs(tm_coef[1]/tm_coef[0]))
    md_fol=pd.DataFrame([fol],columns=['GM(1,1)模型公式'])
    report.table(md_fol)

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
    report.p('建立GM(1,1)模型，得到的预测值与实际值对比图如下图所示。')
    plt.figure(figsize=(6.0, 4.0))
    plt.style.use('ggplot')
    plt.plot(time, timeseries, marker='*', label='origin')
    plt.plot(time, ts_pred[:len(time)], marker='.', alpha=0.7,label='fit')
    plt.legend()
    plt.title('Fitting of GM(1,1) Model')
    plt.xlabel('date')
    plt.ylabel(timeseries.columns.values[0])
    plt.tight_layout()
    plt.savefig('overview.png')
    report.image('overview.png')

    '''
    模型预测情况
    '''
    pre_index=pd.date_range(start=time[params['time']][len(time)-1],periods=n,freq=time[params['time']][len(time)-1]-time[params['time']][len(time)-2])
    report.h3('模型预测情况')
    report.p('设置预测周期数为'+str(n)+'，得到的预测值如下图所示。')
    pre_data=pd.DataFrame({timeseries.columns.values[0]+'_preValue':ts_pred[len(time):]})
    pre_data[timeseries.columns.values[0]+'_preValue']=pre_data.apply(lambda x:'%.2f'%x[timeseries.columns.values[0]+'_preValue'],axis=1)
    pre_data.index=pre_index
    plt.figure(figsize=(6.0, 4.0))
    plt.style.use('ggplot')
    plt.title('Prediction of GM(1,1) Model')
    plt.plot(time,timeseries, '.-', label='origin')
    plt.plot(pre_index,ts_pred[len(time):], '.-', label='prediction')
    plt.legend()
    plt.xlabel('date')
    plt.ylabel(timeseries.columns.values[0])
    plt.tight_layout()
    plt.savefig('pre_view.png')
    report.image('pre_view.png')

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
