import xml.etree.ElementTree as ET

xmlfile = './VOC2007/Annotations/'
jpgpath = './VOC2007/JPEGImages/'

def Need_Augment(path):
    num_yel = 0

    # 获取 XML 文档对象 ElementTree
    tree = ET.parse(path)
    # 获取 XML 文档对象的根结点 Element
    root = tree.getroot()

    # 递归查找所有的 neighbor 子结点
    # 遍历xml文档的第二层
    for child in root:
        for children in child:
            # 第三层节点的标签名称和属性
            if(children.tag == "name"):
                # 判断是否为Yellow，是则加1
                if(children.text == "Yellow"):
                    num_yel = num_yel + 1
    print("num_yel的值为",num_yel)

    if num_yel >= 2:
        return False
    else:
        return True

if __name__ == "__main__":

    IMG_DIR = "./VOC2007/JPEGImages/"
    XML_DIR = "./VOC2007/Annotations/"


    AUG_XML_DIR = "./My_Dataset/Annotations"  # 存储增强后的XML文件夹路径
    AUG_IMG_DIR = "./My_Dataset/JPEGImages"  # 存储增强后的影像文件夹路径

    xml_path = XML_DIR + "tagbike6.xml"

    if(Need_Augment(xml_path)):
        print("需要增强")
