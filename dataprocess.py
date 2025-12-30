import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from wordcloud import WordCloud
import jieba
import jieba.analyse
from collections import Counter
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import json
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class DataAnalyzer:
    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.all_data = pd.DataFrame()
        self.university_location_dict = {}
        self.load_university_locations()
        self.load_data()
    
    def load_data(self):
        for root, dirs, files in os.walk(self.data_folder):
            for file in files:
                if file.endswith('.csv'):
                    file_path = os.path.join(root, file)
                    df = pd.read_csv(file_path, header=1)
                    df['年份'] = os.path.basename(root)
                    self.all_data = pd.concat([self.all_data, df], ignore_index=True)
        
        self.all_data['年份'] = self.all_data['年份'].astype(int)
        if '学校名称' in self.all_data.columns and '项目名称' in self.all_data.columns:
            # 严格过滤掉所有包含'学校名称'的无效条目
            valid_data = self.all_data[self.all_data['学校名称'] != '学校名称']
            valid_data = valid_data[valid_data['学校名称'].str.strip() != '学校名称']
            valid_data = valid_data[~valid_data['学校名称'].str.contains('学校名称', na=False)]
            self.all_data = valid_data.dropna(subset=['学校名称', '项目名称'])
            self.all_data['学校名称'] = self.all_data['学校名称'].str.strip()
            self.all_data['项目名称'] = self.all_data['项目名称'].str.strip()
        
        if '项目类别' in self.all_data.columns:
            category_counts = self.all_data['项目类别'].value_counts()
            valid_categories = category_counts[category_counts > 1].index
            self.all_data = self.all_data[self.all_data['项目类别'].isin(valid_categories)]
        print(f"加载数据完成，共 {len(self.all_data)} 条记录")
    
    def load_university_locations(self):
        # 读取全国大学名单Excel文件
        university_df = pd.read_excel('全国大学名单（教育部）(2).xls', header=1)
            
        # 重命名列名以便于处理
        university_df.columns = ['序号', '学校名称', '学校标识码', '主管部门', '所在地']
            
        # 创建学校名称到所在地的映射字典
        self.university_location_dict = dict(zip(university_df['学校名称'].str.strip(), university_df['所在地'].str.strip()))
            
        print(f"加载大学位置信息完成，共 {len(self.university_location_dict)} 所大学")
    
    def geographic_distribution_heatmap(self):
        universities = self.all_data['学校名称'].value_counts().head(30)
        
        plt.figure(figsize=(14, 8))
        plt.barh(range(len(universities)), universities.values)
        plt.yticks(range(len(universities)), universities.index)
        plt.xlabel('立项数量')
        plt.ylabel('高校名称')
        plt.title('高校立项数量分布（Top 30）')
        plt.tight_layout()
        plt.savefig('C:\\Users\\86151\\Desktop\\数据挖掘\\output\\高校立项分布.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("高校立项分布图已保存")
    
    def geographic_distribution_map(self):
        # 创建城市到省份的映射表
        city_to_province = {
            # 北京
            '北京': '北京',
            # 上海
            '上海': '上海',
            # 天津
            '天津': '天津',
            # 重庆
            '重庆': '重庆',
            # 河北
            '石家庄': '河北', '唐山': '河北', '秦皇岛': '河北', '邯郸': '河北', '邢台': '河北', '保定': '河北', '张家口': '河北', '承德': '河北', '沧州': '河北', '廊坊': '河北', '衡水': '河北',
            # 山西
            '太原': '山西', '大同': '山西', '阳泉': '山西', '长治': '山西', '晋城': '山西', '朔州': '山西', '晋中': '山西', '运城': '山西', '忻州': '山西', '临汾': '山西', '吕梁': '山西',
            # 内蒙古
            '呼和浩特': '内蒙古', '包头': '内蒙古', '乌海': '内蒙古', '赤峰': '内蒙古', '通辽': '内蒙古', '鄂尔多斯': '内蒙古', '呼伦贝尔': '内蒙古', '巴彦淖尔': '内蒙古', '乌兰察布': '内蒙古', '兴安盟': '内蒙古', '锡林郭勒盟': '内蒙古', '阿拉善盟': '内蒙古', '锡林浩特': '内蒙古', '满洲里': '内蒙古', '二连浩特': '内蒙古',
            # 辽宁
            '沈阳': '辽宁', '大连': '辽宁', '鞍山': '辽宁', '抚顺': '辽宁', '本溪': '辽宁', '丹东': '辽宁', '锦州': '辽宁', '营口': '辽宁', '阜新': '辽宁', '辽阳': '辽宁', '盘锦': '辽宁', '铁岭': '辽宁', '朝阳': '辽宁', '葫芦岛': '辽宁',
            # 吉林
            '长春': '吉林', '吉林': '吉林', '四平': '吉林', '辽源': '吉林', '通化': '吉林', '白山': '吉林', '松原': '吉林', '白城': '吉林', '延边朝鲜族自治州': '吉林',
            # 黑龙江
            '哈尔滨': '黑龙江', '齐齐哈尔': '黑龙江', '鸡西': '黑龙江', '鹤岗': '黑龙江', '双鸭山': '黑龙江', '大庆': '黑龙江', '伊春': '黑龙江', '佳木斯': '黑龙江', '七台河': '黑龙江', '牡丹江': '黑龙江', '黑河': '黑龙江', '绥化': '黑龙江', '大兴安岭地区': '黑龙江',
            # 江苏
            '南京': '江苏', '无锡': '江苏', '徐州': '江苏', '常州': '江苏', '苏州': '江苏', '南通': '江苏', '连云港': '江苏', '淮安': '江苏', '盐城': '江苏', '扬州': '江苏', '镇江': '江苏', '泰州': '江苏', '宿迁': '江苏',
            # 浙江
            '杭州': '浙江', '宁波': '浙江', '温州': '浙江', '嘉兴': '浙江', '湖州': '浙江', '绍兴': '浙江', '金华': '浙江', '衢州': '浙江', '舟山': '浙江', '台州': '浙江', '丽水': '浙江',
            # 安徽
            '合肥': '安徽', '芜湖': '安徽', '蚌埠': '安徽', '淮南': '安徽', '马鞍山': '安徽', '淮北': '安徽', '铜陵': '安徽', '安庆': '安徽', '黄山': '安徽', '滁州': '安徽', '阜阳': '安徽', '宿州': '安徽', '六安': '安徽', '亳州': '安徽', '池州': '安徽', '宣城': '安徽',
            # 福建
            '福州': '福建', '厦门': '福建', '莆田': '福建', '三明': '福建', '泉州': '福建', '漳州': '福建', '南平': '福建', '龙岩': '福建', '宁德': '福建',
            # 江西
            '南昌': '江西', '景德镇': '江西', '萍乡': '江西', '九江': '江西', '新余': '江西', '鹰潭': '江西', '赣州': '江西', '吉安': '江西', '宜春': '江西', '抚州': '江西', '上饶': '江西',
            # 山东
            '济南': '山东', '青岛': '山东', '淄博': '山东', '枣庄': '山东', '东营': '山东', '烟台': '山东', '潍坊': '山东', '济宁': '山东', '泰安': '山东', '威海': '山东', '日照': '山东', '临沂': '山东', '德州': '山东', '聊城': '山东', '滨州': '山东', '菏泽': '山东',
            # 河南
            '郑州': '河南', '开封': '河南', '洛阳': '河南', '平顶山': '河南', '安阳': '河南', '鹤壁': '河南', '新乡': '河南', '焦作': '河南', '濮阳': '河南', '许昌': '河南', '漯河': '河南', '三门峡': '河南', '南阳': '河南', '商丘': '河南', '信阳': '河南', '周口': '河南', '驻马店': '河南',
            # 湖北
            '武汉': '湖北', '黄石': '湖北', '十堰': '湖北', '宜昌': '湖北', '襄阳': '湖北', '鄂州': '湖北', '荆门': '湖北', '孝感': '湖北', '荆州': '湖北', '黄冈': '湖北', '咸宁': '湖北', '随州': '湖北', '恩施土家族苗族自治州': '湖北',
            # 湖南
            '长沙': '湖南', '株洲': '湖南', '湘潭': '湖南', '衡阳': '湖南', '邵阳': '湖南', '岳阳': '湖南', '常德': '湖南', '张家界': '湖南', '益阳': '湖南', '郴州': '湖南', '永州': '湖南', '怀化': '湖南', '娄底': '湖南', '湘西土家族苗族自治州': '湖南',
            # 广东
            '广州': '广东', '深圳': '广东', '珠海': '广东', '汕头': '广东', '佛山': '广东', '韶关': '广东', '湛江': '广东', '肇庆': '广东', '江门': '广东', '茂名': '广东', '惠州': '广东', '梅州': '广东', '汕尾': '广东', '河源': '广东', '阳江': '广东', '清远': '广东', '东莞': '广东', '中山': '广东', '潮州': '广东', '揭阳': '广东', '云浮': '广东',
            # 广西
            '南宁': '广西', '柳州': '广西', '桂林': '广西', '梧州': '广西', '北海': '广西', '防城港': '广西', '钦州': '广西', '贵港': '广西', '玉林': '广西', '百色': '广西', '贺州': '广西', '河池': '广西', '来宾': '广西', '崇左': '广西',
            # 海南
            '海口': '海南', '三亚': '海南', '三沙': '海南', '儋州': '海南',
            # 四川
            '成都': '四川', '自贡': '四川', '攀枝花': '四川', '泸州': '四川', '德阳': '四川', '绵阳': '四川', '广元': '四川', '遂宁': '四川', '内江': '四川', '乐山': '四川', '南充': '四川', '眉山': '四川', '宜宾': '四川', '广安': '四川', '达州': '四川', '雅安': '四川', '巴中': '四川', '资阳': '四川', '阿坝藏族羌族自治州': '四川', '甘孜藏族自治州': '四川', '凉山彝族自治州': '四川',
            # 贵州
            '贵阳': '贵州', '六盘水': '贵州', '遵义': '贵州', '安顺': '贵州', '毕节': '贵州', '铜仁': '贵州', '黔西南布依族苗族自治州': '贵州', '黔东南苗族侗族自治州': '贵州', '黔南布依族苗族自治州': '贵州',
            # 云南
            '昆明': '云南', '曲靖': '云南', '玉溪': '云南', '保山': '云南', '丽江': '云南', '普洱': '云南', '临沧': '云南', '楚雄彝族自治州': '云南', '红河哈尼族彝族自治州': '云南', '文山壮族苗族自治州': '云南', '西双版纳傣族自治州': '云南', '大理白族自治州': '云南', '德宏傣族景颇族自治州': '云南', '怒江傈僳族自治州': '云南', '迪庆藏族自治州': '云南',
            # 西藏
            '拉萨': '西藏', '日喀则': '西藏', '昌都': '西藏', '林芝': '西藏', '山南': '西藏', '那曲': '西藏', '阿里': '西藏',
            # 陕西
            '西安': '陕西', '铜川': '陕西', '宝鸡': '陕西', '咸阳': '陕西', '渭南': '陕西', '延安': '陕西', '汉中': '陕西', '榆林': '陕西', '安康': '陕西', '商洛': '陕西',
            # 甘肃
            '兰州': '甘肃', '嘉峪关': '甘肃', '金昌': '甘肃', '白银': '甘肃', '天水': '甘肃', '武威': '甘肃', '张掖': '甘肃', '平凉': '甘肃', '酒泉': '甘肃', '庆阳': '甘肃', '定西': '甘肃', '陇南': '甘肃', '临夏回族自治州': '甘肃', '甘南藏族自治州': '甘肃',
            # 青海
            '西宁': '青海', '海东': '青海', '海北藏族自治州': '青海', '黄南藏族自治州': '青海', '海南藏族自治州': '青海', '果洛藏族自治州': '青海', '玉树藏族自治州': '青海', '海西蒙古族藏族自治州': '青海',
            # 宁夏
            '银川': '宁夏', '石嘴山': '宁夏', '吴忠': '宁夏', '固原': '宁夏', '中卫': '宁夏',
            # 新疆
            '乌鲁木齐': '新疆', '克拉玛依': '新疆', '吐鲁番': '新疆', '哈密': '新疆', '昌吉回族自治州': '新疆', '博尔塔拉蒙古自治州': '新疆', '巴音郭楞蒙古自治州': '新疆', '阿克苏地区': '新疆', '克孜勒苏柯尔克孜自治州': '新疆', '喀什地区': '新疆', '和田地区': '新疆', '伊犁哈萨克自治州': '新疆', '塔城地区': '新疆', '阿勒泰地区': '新疆', '石河子': '新疆', '阿拉尔': '新疆', '图木舒克': '新疆', '五家渠': '新疆', '北屯': '新疆', '铁门关': '新疆', '双河': '新疆', '可克达拉': '新疆', '昆玉': '新疆', '胡杨河': '新疆', '新星': '新疆'
        }
        
        # 统计每个省份的立项数量
        province_counts = {}
        for university in self.all_data['学校名称']:
            university = university.strip()
            if university in self.university_location_dict:
                city = self.university_location_dict[university]
                # 去除"市"后缀
                if city.endswith('市'):
                    city = city[:-1]
                
                # 将城市映射到省份
                province = city_to_province.get(city, '未知')
                
                if province != '未知':
                    if province in province_counts:
                        province_counts[province] += 1
                    else:
                        province_counts[province] = 1
                else:
                    # 尝试直接使用城市作为省份（处理特殊情况）
                    if city in province_counts:
                        province_counts[city] += 1
                    else:
                        province_counts[city] = 1
        
        # 将结果转换为DataFrame
        province_df = pd.DataFrame(list(province_counts.items()), columns=['省份', '立项数量'])
        province_df = province_df.sort_values(by='立项数量', ascending=False)
        
        print("各省份立项数量统计:")
        print(province_df)
        
        # 创建柱状图图形
        plt.figure(figsize=(12, 10))
        ax2 = plt.gca()
        
        bars = ax2.barh(range(len(province_df)), province_df['立项数量'], 
                       color=plt.cm.Reds(np.linspace(0.3, 1, len(province_df))), 
                       height=0.8, edgecolor='black', linewidth=0.5)
            
        # 设置Y轴标签
        ax2.set_yticks(range(len(province_df)))
        ax2.set_yticklabels(province_df['省份'], fontsize=12, fontweight='bold')
            
        # 添加排名数字
        for i in range(len(province_df)):
            ax2.text(-10, i, f'{i+1}', va='center', ha='right', fontsize=12, fontweight='bold', color='gray')
            
        ax2.set_xlabel('立项数量', fontsize=14, fontweight='bold')
        ax2.set_ylabel('省份', fontsize=14, fontweight='bold')
        ax2.set_title('各省份高校立项数量排名', fontsize=20, fontweight='bold')
            
        # 优化网格线
        ax2.grid(axis='x', alpha=0.3, linestyle='--')
            
        # 在柱状图上添加数值标签
        for i, (province, count) in enumerate(zip(province_df['省份'], province_df['立项数量'])):
            ax2.text(count + 5, i, f'{count}', va='center', fontsize=11, fontweight='bold')
            
        # 调整坐标轴范围
        ax2.set_xlim(0, max(province_df['立项数量']) * 1.1)
            
        # 优化整体布局
        ax2.tick_params(axis='y', which='major', pad=20)
        
        plt.tight_layout()
        plt.savefig(r'C:\Users\86151\Desktop\数据挖掘\output\高校立项数量排名柱状图.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("高校立项数量排名柱状图已保存")
    
    def time_series_analysis(self):
        yearly_counts = self.all_data.groupby('年份').size()
        
        plt.figure(figsize=(12, 6))
        plt.plot(yearly_counts.index, yearly_counts.values, marker='o', linewidth=2)
        plt.xlabel('年份')
        plt.ylabel('立项数量')
        plt.title('教育部人文社会科学研究项目逐年趋势')
        plt.grid(True, alpha=0.3)
        plt.subplots_adjust(top=0.9)
        # 设置年份刻度为连续1年间隔
        plt.xticks(range(min(yearly_counts.index), max(yearly_counts.index)+1, 1))
        plt.savefig('C:\\Users\\86151\\Desktop\\数据挖掘\\output\\逐年趋势图.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("逐年趋势图已保存")
    
    def wordcloud_analysis(self):
        all_text = ' '.join(self.all_data['项目名称'].astype(str))
        
        words = jieba.cut(all_text)
        word_list = [word for word in words if len(word) > 1]
        
        word_freq = Counter(word_list)
        
        wordcloud = WordCloud(
            font_path='C:\\Windows\\Fonts\\simhei.ttf',
            width=1200,
            height=800,
            background_color='white',
            max_words=200
        ).generate_from_frequencies(word_freq)
        
        wordcloud.to_file('C:\\Users\\86151\\Desktop\\数据挖掘\\output\\词云图.png')
        print("词云图已保存")
        
        return word_freq
    
    def keyword_trend_analysis(self, word_freq):
        recent_years = [2024, 2025]
        recent_data = self.all_data[self.all_data['年份'].isin(recent_years)]
        recent_text = ' '.join(recent_data['项目名称'].astype(str))
        recent_words = jieba.cut(recent_text)
        recent_word_list = [word for word in recent_words if len(word) > 1]
        recent_word_freq = Counter(recent_word_list)
        print(f"调试信息：近期Top 10关键词: {recent_word_freq.most_common(10)}")
        
        emerging_keywords = []
        for word, count in recent_word_freq.most_common(50):
            if word not in word_freq or word_freq[word] < count * 0.5:
                emerging_keywords.append((word, count))
        
        if emerging_keywords:
            keywords_df = pd.DataFrame(emerging_keywords[:20], columns=['关键词', '出现次数'])
            
            plt.figure(figsize=(12, 8))
            plt.barh(range(len(keywords_df)), keywords_df['出现次数'])
            plt.yticks(range(len(keywords_df)), keywords_df['关键词'])
            plt.xlabel('出现次数')
            plt.ylabel('关键词')
            plt.title('近期（2024-2025）新兴关键词Top 20')
            plt.tight_layout()
            plt.savefig(r'C:\Users\86151\Desktop\数据挖掘\output\新兴关键词.png', dpi=300, bbox_inches='tight')
            plt.show()
            print("新兴关键词图已保存")
        
        return emerging_keywords
    
    def university_clustering(self):
        university_stats = self.all_data.groupby('学校名称').agg({
            '项目名称': 'count',
            '项目类别': lambda x: len(set(x)),
            '学科门类': lambda x: len(set(x))
        }).reset_index()
        university_stats.columns = ['学校名称', '项目总数', '项目类别数', '学科门类数']
        
        scaler = StandardScaler()
        features = scaler.fit_transform(university_stats[['项目总数', '项目类别数', '学科门类数']])
        
        kmeans = KMeans(n_clusters=4, random_state=42)
        university_stats['聚类'] = kmeans.fit_predict(features)
        
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(university_stats['项目总数'], university_stats['学科门类数'], 
                             c=university_stats['聚类'], cmap='viridis', s=100, alpha=0.6)
        plt.xlabel('项目总数')
        plt.ylabel('学科门类数')
        plt.title('高校聚类分析')
        plt.colorbar(scatter, label='聚类类别')
        plt.tight_layout()
        plt.savefig('C:\\Users\\86151\\Desktop\\数据挖掘\\output\\高校聚类分析.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("高校聚类分析图已保存")
        
        return university_stats
    

    def collaboration_network(self):
        applicant_university = self.all_data.groupby('申请人')['学校名称'].apply(list).to_dict()
        
        G = nx.Graph()
        
        for applicant, universities in applicant_university.items():
            if len(universities) > 1:
                for i in range(len(universities)):
                    for j in range(i+1, len(universities)):
                        u1, u2 = universities[i], universities[j]
                        if G.has_edge(u1, u2):
                            G[u1][u2]['weight'] += 1
                        else:
                            G.add_edge(u1, u2, weight=1)
        
        top_universities = self.all_data['学校名称'].value_counts().head(15).index
        G_sub = nx.Graph()
        
        for u, v in G.edges():
            if u in top_universities and v in top_universities and G[u][v]['weight'] >= 3:
                G_sub.add_edge(u, v, weight=G[u][v]['weight'])
        
        if len(G_sub.nodes()) == 0:
            print("没有找到符合条件的合作关系")
            return G
        
        degrees = dict(G_sub.degree())
        max_degree = max(degrees.values()) if degrees else 1
        node_sizes = [degrees[node] * 300 + 500 for node in G_sub.nodes()]

        cmap = plt.cm.viridis  
        node_colors = [cmap(degrees[node]/max_degree) for node in G_sub.nodes()]

        plt.figure(figsize=(24, 20), facecolor='#f8f9fa')
        
        pos = nx.kamada_kawai_layout(G_sub, scale=2.5) 
        
        edges = G_sub.edges()
        weights = [G_sub[u][v]['weight'] for u, v in edges]
        max_weight = max(weights) if weights else 1

        edge_colors = ['#6c757d' for w in weights]  
        edge_widths = [w*2.5 for w in weights] 

        nx.draw_networkx_nodes(G_sub, pos, node_size=node_sizes, node_color=node_colors, 
                               alpha=1.0, linewidths=2, edgecolors='#343a40', 
                               node_shape='o')

        nx.draw_networkx_nodes(G_sub, pos, node_size=[s+30 for s in node_sizes], 
                               node_color='black', alpha=0.15, linewidths=0)
        
        nx.draw_networkx_edges(G_sub, pos, width=edge_widths, 
                               edge_color=edge_colors, alpha=0.7, 
                               style='solid', 
                               connectionstyle='arc3,rad=0.15')  
        

        nx.draw_networkx_labels(G_sub, pos, font_size=14, font_family='SimHei', 
                                font_weight='bold', font_color='#2c3e50',
                                bbox=dict(facecolor='white', edgecolor='#6c757d', 
                                         alpha=0.95, pad=0.6, boxstyle='round,pad=0.4'))

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=max_degree))
        sm.set_array([])
        ax = plt.gca()
        cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label('节点度数（连接数）', fontsize=12, fontweight='bold')
        cbar.ax.tick_params(labelsize=10)
        
        plt.title('学术合作网络可视化\n（Top 15高校间重要合作关系，节点大小表示连接数）', 
                 fontsize=26, pad=60, fontweight='bold', color='#2c3e50')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('C:\\Users\\86151\\Desktop\\数据挖掘\\output\\学术合作网络.png', 
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        print("学术合作网络图已保存")
        
        return G
    
    def category_distribution(self):
        category_counts = self.all_data['项目类别'].value_counts()
        total = sum(category_counts.values)
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8']
        explode = [0.05] * len(category_counts)
        
        plt.figure(figsize=(16, 12), facecolor='#f8f9fa')
        wedges, texts, autotexts = plt.pie(category_counts.values, 
                                            labels=category_counts.index, 
                                            autopct=lambda p: f'{p:.1f}%\n({int(p*total/100)})',
                                            startangle=90, 
                                            colors=colors[:len(category_counts)],
                                            explode=explode,
                                            shadow=True,
                                            textprops={'fontsize': 12, 'fontweight': 'bold'},
                                            labeldistance=1.1)

        for i, (wedge, text, count) in enumerate(zip(wedges, texts, category_counts.values)):
            percentage = (count / total) * 100
            if percentage < 5:  
                ang = (wedge.theta2 - wedge.theta1)/2. + wedge.theta1
                y = np.sin(np.deg2rad(ang))
                x = np.cos(np.deg2rad(ang))

                horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]

                text.set_position((x*1.8, y*1.8))
                text.set_horizontalalignment(horizontalalignment)
                plt.annotate('', xy=(x*1.1, y*1.1), xytext=(x*1.7, y*1.7),
                            arrowprops=dict(arrowstyle="-", color='#2c3e50', linewidth=1.5))
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(10)
        
        plt.title('项目类别分布\n（总数：%d）' % total, 
                 fontsize=20, pad=40, fontweight='bold', color='#2c3e50')
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig('C:\\Users\\86151\\Desktop\\数据挖掘\\output\\项目类别分布.png', 
                    dpi=300, bbox_inches='tight', facecolor='#f8f9fa')
        plt.close()
        print("项目类别分布图已保存")
    
    def category_bar_distribution(self):
        category_counts = self.all_data['项目类别'].value_counts()
        total = sum(category_counts.values)
        
        plt.figure(figsize=(16, 10), facecolor='#f8f9fa')

        bars = plt.bar(category_counts.index, category_counts.values, 
                      color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8'][:len(category_counts)])

        for bar in bars:
            height = bar.get_height()
            percentage = (height / total) * 100
            plt.text(bar.get_x() + bar.get_width()/2., height + 50,
                    f'{height}\n({percentage:.1f}%)',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        plt.title('项目类别分布\n（总数：%d）' % total, 
                 fontsize=22, pad=40, fontweight='bold', color='#2c3e50')
        plt.xlabel('项目类别', fontsize=16, fontweight='bold')
        plt.ylabel('数量', fontsize=16, fontweight='bold')
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.tight_layout()
        plt.savefig('C:\\Users\\86151\\Desktop\\数据挖掘\\output\\项目类别分布柱状图.png', 
                    dpi=300, bbox_inches='tight', facecolor='#f8f9fa')
        plt.close()
        print("项目类别分布柱状图已保存")
    
    def discipline_year_trend(self):
        top_disciplines = self.all_data['学科门类'].value_counts().head(10).index
        trend_data = self.all_data[self.all_data['学科门类'].isin(top_disciplines)]
        discipline_year = trend_data.groupby(['年份', '学科门类']).size().unstack(fill_value=0)
        
        colors = plt.cm.tab10.colors
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
        line_styles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--'] 
        
        plt.figure(figsize=(16, 10), facecolor='#f8f9fa')
        ax = plt.gca()
        
        for i, discipline in enumerate(discipline_year.columns):
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]
            linestyle = line_styles[i % len(line_styles)]
            
            plt.plot(discipline_year.index, discipline_year[discipline], 
                     marker=marker, linestyle=linestyle, linewidth=3, markersize=8, 
                     color=color, label=discipline)
        
        plt.xlabel('年份', fontsize=16, fontweight='bold', color='#2c3e50')
        plt.ylabel('立项数量', fontsize=16, fontweight='bold', color='#2c3e50')
        plt.title('主要学科门类逐年趋势', fontsize=20, fontweight='bold', color='#2c3e50', pad=30)
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=12, framealpha=0.95, 
                  facecolor='white', edgecolor='#6c757d', borderpad=1, labelspacing=1.2)
        
        plt.grid(True, linestyle='--', alpha=0.4, color='#bdc3c7')
        
        plt.xticks(range(min(discipline_year.index), max(discipline_year.index)+1, 1), 
                  rotation=45, ha='right', fontsize=11, fontweight='medium')
        plt.yticks(fontsize=11, fontweight='medium')
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#6c757d')
        ax.spines['bottom'].set_color('#6c757d')
        ax.tick_params(axis='both', colors='#2c3e50', width=2)
        
        plt.tight_layout()
        plt.savefig(r'C:\Users\86151\Desktop\数据挖掘\output\学科门类趋势.png', dpi=300, bbox_inches='tight', facecolor='#f8f9fa')
        plt.show()
        print("学科门类趋势图已保存")
    
    def run_all_analysis(self):
        print("\n1. 地理分布分析")
        self.geographic_distribution_heatmap()
        
        print("\n3. 时间序列分析")
        self.time_series_analysis()
        
        print("\n4. 文本分析与词云图")
        word_freq = self.wordcloud_analysis()
        
        print("\n5. 新兴关键词分析")
        self.keyword_trend_analysis(word_freq)
        
        print("\n6. 聚类分析")
        self.university_clustering()
        
        print("\n7. 学术合作网络分析")
        self.collaboration_network()
        
        print("\n8. 项目类别分布")
        self.category_distribution()
        self.category_bar_distribution()
        
        print("\n9. 学科门类趋势")
        self.discipline_year_trend()

if __name__ == '__main__':
    data_folder = r'C:\Users\86151\Desktop\数据挖掘\output\PDF_files'
    print(f"数据文件夹: {data_folder}")
    analyzer = DataAnalyzer(data_folder)
    analyzer.run_all_analysis()