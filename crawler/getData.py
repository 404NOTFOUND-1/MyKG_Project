import os.path

from DrissionPage import ChromiumPage
from tqdm import tqdm

urls = ['https://zhuanlan.zhihu.com/p/411459773',
        'https://zhuanlan.zhihu.com/p/499836463',
        'https://zhuanlan.zhihu.com/p/348577547'
        ]
path = 'output.txt'
if os.path.exists(path):
    os.remove(path)

page = ChromiumPage()
for url in tqdm(urls, desc='爬取中'):
    # 获取页面
    page.get(url)
    # 查找元素
    ele = page.ele('.RichText ztext Post-RichText css-jflero')
    # 查找子元素并剔除空行
    All = ele.children('tag:p@@data-pid')
    # 保存文本数据
    with open(path, 'a', encoding='utf8') as f:
        for line in All:
            f.write(line.text + '\n')
