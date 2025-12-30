import os
import re
import requests
from bs4 import BeautifulSoup
import logging

# 配置日志
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MoePdfCrawler:
    def __init__(self):
        # 教育部网站配置
        self.moe_base_url = 'http://www.moe.gov.cn'
        self.moe_target_url = 'http://www.moe.gov.cn/s78/A13/tongzhi/'
        
        # sinoss.net网站配置
        self.sinoss_base_url = 'https://www.sinoss.net'
        self.sinoss_target_url = 'https://www.sinoss.net/gl/tzgg/'
        
        # 输出目录配置
        self.output_dir = 'output'
        self.pdf_dir = os.path.join(self.output_dir, 'PDF_files')
        
        # 请求头配置
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0'
        }
        
        # 创建输出目录
        os.makedirs(self.pdf_dir, exist_ok=True)
    
    def get_page_content(self, url):
        response = requests.get(url, headers=self.headers, timeout=10)
        response.raise_for_status()
        response.encoding = response.apparent_encoding
        return response.text
    
    def find_notice_pages(self, start_year=2010, end_year=2025):
        logger.info(f"查找{start_year}-{end_year}年间的目标通知页面")
        
        all_notice_urls = []
        
        # 1. 从教育部网站查找通知页面
        logger.info("\n从教育部网站查找通知页面")
        moe_notice_urls = self._find_moe_notice_pages(start_year, end_year)
        all_notice_urls.extend(moe_notice_urls)
        
        # 2. 从sinoss.net网站查找通知页面
        logger.info("\n从社科网查找通知页面")
        sinoss_notice_urls = self._find_sinoss_notice_pages(start_year, end_year)
        all_notice_urls.extend(sinoss_notice_urls)
        
        # 去重，避免同一通知被多次抓取
        unique_notice_urls = []
        seen_urls = set()
        for notice in all_notice_urls:
            if notice["url"] not in seen_urls:
                seen_urls.add(notice["url"])
                unique_notice_urls.append(notice)
        
        if not unique_notice_urls:
            logger.error("未找到任何目标通知页面")
        else:
            # 按年份排序
            unique_notice_urls.sort(key=lambda x: x["year"])
            logger.info(f"\n共找到{len(unique_notice_urls)}个目标通知页面")
            
        return unique_notice_urls
    
    def _find_moe_notice_pages(self, start_year=2010, end_year=2025):
        notice_urls = []
        
        # 分页参数
        wcmid = 2665
        wasid = 254874
        page_size = 20
        total_records = 769
        max_static_pages = 25
        
        # 计算总页数
        total_pages = (total_records + page_size - 1) // page_size
        logger.info(f"教育部网站总记录数: {total_records}, 每页记录数: {page_size}, 总页数: {total_pages}")
        
        # 爬取前25页的静态HTML页面
        logger.info("爬取教育部网站前25页的静态HTML页面")
        for page_num in range(0, min(max_static_pages, total_pages)):
            if page_num == 0:
                # 第一页是index.html
                page_url = self.moe_target_url
            else:
                # 其他静态页面是index_{page_num-1}.html
                page_url = f'{self.moe_target_url}index_{page_num-1}.html'
            
            logger.info(f"正在爬取教育部静态页面: {page_url}")
            page_content = self.get_page_content(page_url)
            if not page_content:
                continue
        
            extracted_links = self._extract_moe_notice_links_from_page(page_content, start_year, end_year)
            notice_urls.extend(extracted_links)
        
        # 爬取第25页之后的页面（通过AJAX接口）
        if total_pages > max_static_pages:
            logger.info(f"开始爬取教育部网站第{max_static_pages}页之后的动态页面...")
            for page_num in range(max_static_pages, total_pages + 1):
                # AJAX接口URL
                ajax_url = f'{self.moe_base_url}/was5/web/search?channelid={wasid}&chnlid={wcmid}&page={page_num}'
                logger.info(f"正在爬取教育部动态页面: {ajax_url}")
                
                page_content = self.get_page_content(ajax_url)
                if not page_content:
                    continue
                
                # 从页面提取通知链接
                extracted_links = self._extract_moe_notice_links_from_page(page_content, start_year, end_year)
                notice_urls.extend(extracted_links)
        
        logger.info(f"从教育部网站找到{len(notice_urls)}个目标通知页面")
        return notice_urls
    
    def _find_sinoss_notice_pages(self, start_year=2010, end_year=2025):
        notice_urls = []
        
        total_records = 1154  # 从页面分析得到共1154条记录
        records_per_page = 15
        total_pages = (total_records + records_per_page - 1) // records_per_page  # 计算总页数
        
        logger.info(f"社科网总记录数: {total_records}, 总页数: {total_pages}")
        
        # 遍历所有页面
        for page_num in range(1, total_pages + 1):
            # 构建页面URL
            if page_num == 1:
                page_url = self.sinoss_target_url
            else:
                page_url = f'{self.sinoss_target_url}index_{page_num}.shtml'
            
            logger.info(f"正在爬取社科网页面: {page_url}")
            page_content = self.get_page_content(page_url)
            if not page_content:
                continue
            
            # 从页面提取通知链接
            extracted_links = self._extract_sinoss_notice_links_from_page(page_content, start_year, end_year)
            notice_urls.extend(extracted_links)
        
        logger.info(f"从社科网找到{len(notice_urls)}个目标通知页面")
        return notice_urls
    
    def _extract_moe_notice_links_from_page(self, page_content, start_year=2010, end_year=2025):
        extracted_links = []
        
        soup = BeautifulSoup(page_content, 'html.parser')
        
        # 查找所有包含"教育部人文社会科学研究一般项目立项的通知"或"人文社会科学研究一般项目立项的通知"的链接
        notice_pattern = r'[教育部]*人文社会科学研究.*一般项目.*立项.*通知'
        notice_links = soup.find_all('a', string=re.compile(notice_pattern))
        
        if notice_links:
            for link in notice_links:
                notice_url = link.get('href')
                notice_text = link.text.strip()
                
                # 提取年份信息
                year_match = re.search(r'(\d{4})[年度]', notice_text)
                if year_match:
                    year = int(year_match.group(1))
                    # 只保留指定年份范围内的通知
                    if start_year <= year <= end_year:
                        # 处理相对路径
                        if notice_url.startswith('./'):
                            # 特殊处理教育部网站的相对路径格式
                            if notice_url.startswith('./20'):
                                # 如./202512/t20251202_1422099.html 转换为http://www.moe.gov.cn/s78/A13/tongzhi/202512/t20251202_1422099.html
                                notice_url = f'{self.moe_target_url}{notice_url[2:]}'
                            else:
                                notice_url = f'{self.moe_base_url}{notice_url[1:]}'
                        elif not notice_url.startswith('http'):
                            if notice_url.startswith('/'):
                                notice_url = f'{self.moe_base_url}{notice_url}'
                            else:
                                notice_url = f'{self.moe_target_url}{notice_url}'
                        
                        extracted_links.append({"url": notice_url, "year": year, "text": notice_text})
                        logger.info(f"找到{year}年目标通知页面: {notice_url}")
        
        return extracted_links
    
    def _extract_sinoss_notice_links_from_page(self, page_content, start_year=2010, end_year=2025):
        extracted_links = []
        
        soup = BeautifulSoup(page_content, 'html.parser')
        
        # 查找所有包含"人文社会科学研究"或"一般项目"或"立项"的链接
        notice_pattern = r'[教育部]*人文社会科学研究.*一般项目.*立项.*通知'
        notice_links = soup.find_all('a', string=re.compile(notice_pattern))
        
        if notice_links:
            for link in notice_links:
                notice_url = link.get('href')
                notice_text = link.text.strip()
                
                # 提取年份信息
                year_match = re.search(r'(\d{4})[年度]', notice_text)
                if not year_match:
                    # 从链接中的日期提取年份
                    date_match = re.search(r'/c/(\d{4})-\d{2}-\d{2}/', notice_url)
                    if date_match:
                        year = int(date_match.group(1))
                    else:
                        continue
                else:
                    year = int(year_match.group(1))
                
                # 只保留指定年份范围内的通知
                if start_year <= year <= end_year:
                    # 处理相对路径
                    if not notice_url.startswith('http'):
                        notice_url = f'{self.sinoss_base_url}{notice_url}'
                    
                    extracted_links.append({"url": notice_url, "year": year, "text": notice_text})
                    logger.info(f"找到{year}年目标通知页面: {notice_url}")
        
        return extracted_links
    
    def extract_pdf_links(self, page_url):
        logger.info(f"从页面提取PDF链接: {page_url}")
        page_content = self.get_page_content(page_url)
        if not page_content:
            return []
        
        soup = BeautifulSoup(page_content, 'html.parser')
        pdf_links_with_info = []
        
        # 查找所有a标签，提取PDF链接
        all_links = soup.find_all('a')
        logger.info(f"从页面找到 {len(all_links)} 个链接")
        
        # 查找所有可能包含PDF的链接，不区分大小写
        pdf_extensions = ['.pdf', '.PDF']
        
        for link in all_links:
            href = link.get('href')
            text = link.text.strip()
            
            # 打印所有链接信息用于调试
            logger.debug(f"链接: href={href}, text={text}")
            
            # 检查链接是否指向PDF文件（不区分大小写）
            is_pdf = False
            if href:
                for ext in pdf_extensions:
                    if ext in href:
                        is_pdf = True
                        break
            
            if is_pdf:
                # 过滤条件：标题中有"不予立项"的PDF不下载
                if '不予立项' in text:
                    logger.info(f"跳过不予立项的PDF: {text}")
                    continue
                
                # 过滤条件：只下载指定类型的PDF
                allowed_types = ['规划基金', '青年基金', '自筹经费项目', '西部和边疆地区项目', '新疆项目', '西藏项目']
                is_allowed = False
                for pdf_type in allowed_types:
                    if pdf_type in text:
                        is_allowed = True
                        break
                
                if not is_allowed:
                    logger.info(f"跳过非指定类型的PDF: {text}")
                    continue
                
                # 根据页面URL判断是哪个网站的页面
                if 'moe.gov.cn' in page_url:
                    # 教育部网站的PDF链接处理
                    full_url = self._process_moe_pdf_url(href, page_url)
                elif 'sinoss.net' in page_url:
                    # sinoss.net网站的PDF链接处理
                    full_url = self._process_sinoss_pdf_url(href)
                else:
                    # 通用处理
                    if href.startswith('http'):
                        full_url = href
                    elif href.startswith('/'):
                        # 尝试使用两个网站的基础URL
                        if 'moe.gov.cn' in page_url:
                            full_url = f'{self.moe_base_url}{href}'
                        elif 'sinoss.net' in page_url:
                            full_url = f'{self.sinoss_base_url}{href}'
                        else:
                            # 默认使用moe_base_url
                            full_url = f'{self.moe_base_url}{href}'
                    else:
                        # 尝试使用页面URL的目录部分
                        base_url = '/'.join(page_url.split('/')[:-1]) + '/'
                        full_url = base_url + href
                
                # 提取年份信息
                year = 'unknown'

                year_match = re.search(r'(\d{4})[年度]', text)
                if year_match:
                    year = year_match.group(1)
                else:
                    url_year_match = re.search(r'(\d{4})', page_url)
                    if url_year_match:
                        year = url_year_match.group(1)
                    else:
                        pdf_url_year_match = re.search(r'(\d{4})', full_url)
                        if pdf_url_year_match:
                            year = pdf_url_year_match.group(1)
                
                pdf_links_with_info.append({
                    'url': full_url,
                    'title': text,
                    'year': year
                })
                logger.info(f"找到符合条件的PDF: {text}, URL: {full_url}, 年份: {year}")
        
        if not pdf_links_with_info:
            logger.info("未找到直接的PDF链接")
            iframes = soup.find_all('iframe')
            logger.info(f"找到 {len(iframes)} 个iframe")
            
            for iframe in iframes:
                iframe_src = iframe.get('src')
                logger.debug(f"iframe: src={iframe_src}")
                
                if iframe_src:
                    # 检查iframe源是否指向PDF
                    for ext in pdf_extensions:
                        if ext in iframe_src:
                            # 构建完整的iframe源URL
                            if iframe_src.startswith('http'):
                                full_url = iframe_src
                            elif iframe_src.startswith('/'):
                                if 'moe.gov.cn' in page_url:
                                    full_url = f'{self.moe_base_url}{iframe_src}'
                                elif 'sinoss.net' in page_url:
                                    full_url = f'{self.sinoss_base_url}{iframe_src}'
                                else:
                                    full_url = f'{self.moe_base_url}{iframe_src}'
                            else:
                                base_url = '/'.join(page_url.split('/')[:-1]) + '/'
                                full_url = base_url + iframe_src
                            
                            # 提取年份信息
                            year = 'unknown'
                            url_year_match = re.search(r'(\d{4})', page_url)
                            if url_year_match:
                                year = url_year_match.group(1)
                            else:
                                pdf_url_year_match = re.search(r'(\d{4})', full_url)
                                if pdf_url_year_match:
                                    year = pdf_url_year_match.group(1)
                            
                            pdf_links_with_info.append({
                                'url': full_url,
                                'title': f'iframe_PDF_{len(pdf_links_with_info) + 1}',
                                'year': year
                            })
                            logger.info(f"找到iframe中的PDF: URL: {full_url}, 年份: {year}")
                            break
        
        logger.info(f"共找到 {len(pdf_links_with_info)} 个PDF链接")

        return pdf_links_with_info[:4]
    
    def _process_moe_pdf_url(self, href, page_url):
        if href.startswith('./'):
            # 特殊处理教育部网站的相对路径格式
            if href.startswith('./W'):
                page_dir = '/'.join(page_url.split('/')[:-1]) + '/'
                full_url = f'{page_dir}{href[2:]}'
            else:
                # 正常相对路径
                full_url = f'{page_url.rsplit('/', 1)[0]}/{href[2:]}'
        elif href.startswith('http'):
            full_url = href
        elif href.startswith('/'):
            full_url = f'{self.moe_base_url}{href}'
        else:
            full_url = f'{page_url.rsplit('/', 1)[0]}/{href}'
        return full_url
    
    def _process_sinoss_pdf_url(self, href):
        if href.startswith('http'):
            full_url = href
        elif href.startswith('/'):
            full_url = f'{self.sinoss_base_url}{href}'
        else:
            full_url = f'{self.sinoss_base_url}/{href}'
        return full_url
    
    def download_pdf(self, pdf_info):
        pdf_url = pdf_info['url']
        pdf_title = pdf_info['title']
        year = pdf_info['year']
        
        # 创建年份目录
        year_dir = os.path.join(self.pdf_dir, year)
        os.makedirs(year_dir, exist_ok=True)
        
        # 清理文件名
        clean_title = re.sub(r'[\/:*?"<>|]', '_', pdf_title)
        # 检查标题是否已包含PDF扩展名（不区分大小写）
        if not (clean_title.lower().endswith('.pdf')):
            clean_title += '.pdf'
        filename = clean_title
        save_path = os.path.join(year_dir, filename)
        
        # 检查文件是否已存在
        if os.path.exists(save_path):
            logger.info(f"文件已存在，跳过下载: {save_path}")
            return True
        
        logger.info(f"开始下载PDF: {pdf_title}")
        response = requests.get(pdf_url, headers=self.headers, timeout=30, stream=True)
        response.raise_for_status()
            
        # 获取文件大小
        file_size = int(response.headers.get('content-length', 0))
        downloaded_size = 0
            
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)
                        
                    # 打印下载进度
                    if file_size > 0:
                        progress = downloaded_size / file_size * 100
                        logger.info(f"下载进度: {progress:.2f}% - {pdf_title}")
            
        logger.info(f"PDF下载完成: {save_path}")
        return True
    
    def crawl(self, start_year=2010, end_year=2025):
        logger.info(f"执行教育部PDF爬虫，爬取年份范围: {start_year}-{end_year}")
        
        # 1. 找到所有通知页面
        notice_pages = self.find_notice_pages(start_year, end_year)
        
        total_pdfs_downloaded = 0
        
        # 2. 遍历每个通知页面，提取并下载PDF链接
        for notice in notice_pages:
            notice_url = notice["url"]
            notice_year = notice["year"]
            notice_text = notice["text"]
            
            logger.info(f"\n处理 {notice_year} 年的通知: {notice_text}")
            
            # 3. 提取前四个PDF链接
            pdf_links = self.extract_pdf_links(notice_url)
            if not pdf_links:
                logger.error(f"未从页面找到PDF链接: {notice_url}")
                continue
            
            logger.info(f"从页面找到 {len(pdf_links)} 个PDF文件")
            
            # 4. 下载PDF文件
            for i, pdf_info in enumerate(pdf_links, 1):
                logger.info(f"正在下载 {notice_year} 年第 {i}/{len(pdf_links)} 个PDF文件")
                if self.download_pdf(pdf_info):
                    total_pdfs_downloaded += 1
        
        logger.info(f"\n爬虫执行完成，共下载了 {total_pdfs_downloaded} 个PDF文件")

if __name__ == "__main__":
    crawler = MoePdfCrawler()
    crawler.crawl(start_year=2010, end_year=2025)