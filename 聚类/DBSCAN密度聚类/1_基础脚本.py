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
columns
特征
模型训练的特征。请选择数值型数据，如果勾选了非数值类型数据，则会自动过滤，下个组件可能无法获取所有列。

add_col
定义聚类号列名
cluster_id      [默认值]
"""

# 普通参数
"""
eps
半径
0.5
同一领域两个样本的最大距离。 [0,1]之间的double类型

min_samples
领域内最小数
5
领域内最小数目，大于0的整数
"""

# 高级参数
"""
algorithm
NearestNeighbors模块使用的算法
auto
NearestNeighbors模块使用的算法     计算逐点距离并找到最近邻居。     有关详细信息，请参阅NearestNeighbors模块文档。

leaf_size
叶子大小
30
叶子大小传递给BallTree或cKDTree。这会影响速度     构造和查询，以及所需的内存     存储树。最佳值取决于     关于问题的性质。
"""


def execute(conn, inputs, params, outputs, reportFileName):
    #<editable>
    '''
    载入模块
    '''
    import numpy as np
    import pandas as pd
    from sklearn.cluster import DBSCAN
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    import pyh
    import report_utils
    import db_utils
    import warnings
    warnings.filterwarnings('ignore')
    report = report_utils.Report()

    '''
    选择目标数据
    '''
    data_in = db_utils.query(conn, 'select '+ params['columns'] + ' from ' + inputs['data_in'])

    '''
    报告
    '''
    report.h1('层次聚类')
    model_params={}
    model_params['领域内最小样本数目']=params['min_samples']
    model_params['领域大小']=params['eps']
    model_params['叶子大小']=params['leaf_size']
    model_params['NearestNeighbors模块使用的算法']=params['algorithm']
    a=pd.DataFrame([model_params.keys(),model_params.values()]).T
    a.columns=['参数名称','参数值']
    report.h3('模型参数')
    report.p("输出配置的参数以及参数的取值。")
    report.table(a)

    '''
    层次聚类
    '''
    data_in = data_in.select_dtypes(include=['number'])  # 筛选数值型数据
    clst = DBSCAN(eps = float(params['eps']), min_samples = int(params['min_samples']),algorithm=params['algorithm'],leaf_size=int(params['leaf_size']))

    '''
    预测聚类标签并输出
    '''
    fit_label = clst.fit_predict(data_in) + 1
    data_out = np.column_stack((data_in, fit_label))
    columns = data_in.columns
    columns = np.append(columns, params['add_col'])
    data_out = pd.DataFrame(data_out, columns = columns)
    lable_res=pd.Series(fit_label)
    lable_res=lable_res.value_counts()
    lable_res=pd.DataFrame(lable_res)
    lable_res['group']=lable_res.index
    lable_res['group']=lable_res['group'].apply(lambda x:'group'+str(x))
    lable_res=lable_res.reset_index(drop=True)
    fig = plt.figure(figsize=(8,6))
    a=pd.DataFrame([lable_res['group'],lable_res[0]])
    b=[]
    for i in range(len(lable_res)):
        b.append(0.1)
    labels = pd.Series(lable_res['group']).unique()
    fracs = list(lable_res[0])
    plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
    plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
    plt.axes(aspect=1)
    plt.pie(x=fracs,labels=labels,
            explode=tuple(b),
            autopct='%3.1f %%',
            shadow=True, labeldistance=1.1,
            startangle = 180,
            pctdistance = 0.6,
            radius=2.5
           )
    plt.axis('equal')
    plt.title('pie')
    plt.legend(loc=0, bbox_to_anchor=(0.92, 1))  # 图例
    #设置legend的字体大小
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontsize=6)
    plt.savefig("pie.png")
    plt.show()

    '''
    报告
    '''
    report.h3('饼图结果概况')
    string='由饼图可以看出，总共分为'+str(len(lable_res))+'个聚类分群，分别是：'
    for i in range(len(lable_res)):
        if(i<len(lable_res)-1):
            string=string+lable_res.loc[i][1]+"、"
        else:
            string=string+lable_res.loc[i][1]+"。"
    report.p(string)
    for i in range(len(lable_res)):
        report.p(lable_res.loc[i][1]+"的个数为"+str(lable_res.loc[i][0])+"，占比为"+str(np.around(lable_res.loc[i][0]/lable_res[0].sum(),decimals=2)))
    report.image('pie.png')
    out=data_in
    out['label']=fit_label
    tsne = TSNE(n_components=2, learning_rate=100).fit_transform(out)
    fig = plt.figure()
    x=tsne[:,0]
    y=tsne[:,1]
    plt.scatter(x, y, c=out['label'])
    plt.title("scatter\n");
    plt.savefig('scatter.png')
    plt.show()
    report.h3('散点图示例')
    report.p('通过对数据进行降维，在二维空间中展示的聚类结果。')
    report.image('scatter.png')
    report.writeToHtml(reportFileName)

    '''
    将结果写出
    '''
    db_utils.dbWriteTable(conn, outputs['data_out'], data_out)

    #</editable>
