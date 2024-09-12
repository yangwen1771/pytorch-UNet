from PIL import Image


def keep_image_size_open(path, size=(256, 256)):
    img = Image.open(path)                              #读入图片
    temp = max(img.size)                                #根据最长边做mask掩码#?img.size是什么类型的数据
    mask = Image.new('RGB', (temp, temp),(0,0,0))       #'RGB'类型, 正方形长宽(temp, temp),纯黑色(0,0,0)
    mask.paste(img, (0, 0))                             #把图片粘到左上角
    mask = mask.resize(size)                            #统一resize为256
    return mask
def keep_image_size_open_rgb(path, size=(256, 256)):
    img = Image.open(path)
    temp = max(img.size)
    mask = Image.new('RGB', (temp, temp))
    mask.paste(img, (0, 0))
    mask = mask.resize(size)
    return mask
