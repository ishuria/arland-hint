import json
import os
import platform
import random
import signal
from transformers import AutoTokenizer, AutoModel
import readline
from util.db_util import get_difference_it_item_index_id_item_id_mapping, save_llm_answer
from util.file_util import read_file_content_as_string, read_file_content
from util.config_util import DATA_FOLDER, CHATGLM_6B_FOLDER
from util.answer_util import extract_answer_from_str

tokenizer = AutoTokenizer.from_pretrained(CHATGLM_6B_FOLDER, trust_remote_code=True)
model = AutoModel.from_pretrained(CHATGLM_6B_FOLDER, trust_remote_code=True).half().cuda()
model = model.eval()

os_name = platform.system()
clear_command = 'cls' if os_name == 'Windows' else 'clear'
stop_stream = False

if __name__ == "__main__":
    it_item_index_id_mapping, id_item_id_map = get_difference_it_item_index_id_item_id_mapping()
    hints = '''
扫描二维码存在一定的风险;人工智能包括语音识别、字符识别、机器翻译，此外，生活中常见的人工智能技术还有指纹识别、人脸识别、图像识别、自动驾驶技术等。;多媒体数据的压缩属于编码过程，解压缩属于解码过程。多媒体技术有三个显著的特征：集成性、交互性、实时性。实时收看节目体现了多媒体技术实时性的特征。多媒体数据库技术、多媒体通信技术是多媒体技术发展的热点。;通过微信欣赏同学发布的照片体现了信息的共享性;别人的照片和小说不能随便发布或分享，一则违反道德，二则可能触犯法律；购买正版软件只是取得软件的使用权，没有版权，将自己购买并在使用的正版Windows10序列号借给他人激活违反了软件版权；在观影过程中发表“弹幕”可以互相传达各自意见增强交流感;个人博客中发布他人隐私信息、组织用黑客软件攻击网站和破解网银账户密码方法等均属违法行为;数据表可以和字段名相同；删除记录后，自动编号类型字段值不会自动改变;多媒体作品设计的一般过程：需求分析、规划设计、脚本编写。（1）需求分析包括应用需求分析和创作需求分析。（2）规划设计包括结构设计和模块设计。（3）脚本编写的一般过程：先制订脚本大纲，然后编写文字脚本，再编写制作脚本，最后进行媒体元素分解。;“静音”操作不会影响音频时长；单声道音频执行“删除”操作后，删掉选中区域，后面的音频向前靠拢，音频变为\(20\)秒，音频时长改变存储容量改变；由于音频是\(\text{MP3}\)格式，不支持用单声道容量乘声道数来计算音频容量；“插入静音”操作会在选区的前端插入静音。;\(\text{SMTP}\)协议用于发送邮件，从发件人的邮件服务器发送到收件人的邮件服务器；浏览器主要提供\(\text{HTTP}\)协议，其作用主要是传输、编译、解释和执行\(\text{HTML}\)代码；网络下载的文件都有可能存在病毒，建议先查杀再使用；“\(*.\text{mht}\)”格式也属于网页文件格式，是将源代码、网页素材等全部资源封装在一个独立的文件中，可以有图文，也可以有超链接;\(\mathrm{A}\mathrm{S}\mathrm{C}\mathrm{I}\mathrm{I}\)码和汉字编码。观察图中共有\(2\)个字节的内码值不超过\(7\mathrm{F}\)；字符“\(\mathrm{a}\)”的十六进制内码值为\(61\)，字符“\(\mathrm{j}\)”比字符“\(\mathrm{a}\)”大\(9\)，即\(6\mathrm{A}\)；字符“\(\mathrm{z}\)”的十六进制内码为\(7\mathrm{A}\)，将该十六进制数转换为十进制数，即为\(7\ast 1{6}^{1}+10\ast 1{6}^{0}=122\)；字符“母”对应的十六进制内码为“\(\mathrm{C}4\;\mathrm{B}8\)”，将该十六进制数转换为二进制数为“\(11000100\;10111000\)”;人工智能相关知识。人工智能：用机器来完成某些需要人类智慧才能完成的任务。主要有以下两类：（1）模式识别，如指纹识别、语音识别、光学字符识别、手写识别等。（②）机器翻译，利用计算机把一种自然语言变成另一种自然语言，如中英文互译。用指纹识别进行身份认证是人工智能中的模式识别;信息的特征。信息有真也有假，我们需要提高辨别信息真伪的能力。王先生最后被骗了\(2\)万元，说明了信息具有真伪性;\(\text{UltraEdit}\)软件字符编码采用十六进制数表示。如图所示，逗号“，”的内码是\(\text{2C}\)，转换为二进制是\(00101100\)；存储图中字符共需要\(\text{14byte}\)存储空间；“\(2020\)”的内码用十六进制可表示为\(32\;30\;32\;30\)；\(\text{ASCII}\)码最大值为\(\text{7F}\)，图中共有\(6\)个\(\text{ASCII}\)码，是“\(2\mathrm{C}\;20\;32\;30\;31\;39\)”，每个汉字为\(1\)个\(\text{GB2312}\)码，所以共有\(4\)个\(\text{GB2312}\)码;“画图”程序将“风景\(.\text{bmp}\)”另存为“风景\(.\text{jpg}\)”数据进行了有损压缩，数据失真，存储容量减小，不能还原；用\(\text{WinRAR}\)软件生成压缩包“风景\(.\text{rar}\)”，这种压缩是无损压缩，可以进行还原;字符串类型数据必须以单引号或双引号括起来的任意文本;\(\text{Flash}\)库中素材可以有声音、位图、图形、影片剪辑、按钮，其中属于元件的只有图形、影片剪辑、按钮共三种类型;图像信息的采集。要截取计算机屏幕中的一幅画面，可以使用“\(\text{PrintScreen}\)”键、使用数码照相机拍摄、使用专用计算机屏幕截图软件，不可以使用扫描仪;\(\text{Photoshop}\)的实际操作。图3中只有“校园好声音”图层处于选中状态，所以向上拖动此图层时，“欢呼”图层不会一起移动；将图1复制到图2时，会新建图层并粘贴到新图层故B项错误；“欢呼”图层未处于锁定状态，是可以停用其图层样式的；“背景”图层虽然处于锁定状态，但是不影响其滤镜的使用;“文字”图层的补间是虚线，表示补间创建失败。“音乐”图层只有一个空白关键帧。该动画有\(36\)帧，帧频为\(12.0\text{fps}\)，则播放一次需\(3\)秒。“雪花”图层有\(36\)帧，其中只有两个关键帧。;语句“\(\text{select }*\text{ from s_info}\)”没有指定字段和条件，直接打开整张数据表。;\(\text{VB}\)中的运算符和表达式。\(\text{X=4X}\)是代数式而非\(\text{VB}\)表达式;二进制的\(0\)和十六进制的\(0\)末位都是\(0\)，而二进制数\(1000\)末位数是\(0\)，它对应的十六进制数是\(8\)，末位不是\(0\)；所以二进制数末位为\(0\)，该数对应的十六进制数末位不一定是\(0\)；二进制数\(1110001\)转换为十六进制数，采用\(8421\)法，\(01110001=71\)；二进制\(1110\)末位数\(0\)对应的权值是\(2^{0}\)；二进制数\(1101\)转换为十进制数是\(1101=1\text{*}2^{3}+1\text{*}2^{3}+0\text{*}2^{1}+1\text{*}2^{0}=13\)，去掉末位\(1\)，新数\(110=1\text{*}2^{2}+1\text{*}2^{1}+0\text{*}2^{0}=6\)，\(6\)不是\(13\)的\(1/2\);网络分类方法有局域网/广域网、环型网/星型网、有线网/无线网，没有对等网/城域网;\(\text{Access}\)数据库相关知识：“访问时间”的字段类型是“日期/时间”，该表至少有\(5\)个字段，共有\(67\)条记录;信源越权威越可靠，兰州马拉松官方网站是权威部门，从它获取的信息最可靠;当前状态下，不可以删除“图书信息”数据表；“有无光盘”字段为“是/否”类型；不可在“书号”为\(2\)的记录前插入一条新记录;进制数转换。\(\text{15}\)的\(\text{7}\)位二进制数为\(0001111\)，取反后为\(1110000\)，加\(\text{1}\)后为\(1110001\)，设置符号位为\(\text{1}\)，\(-\text{15}\)的\(\text{8}\)位二进制补码为\(11110001\)。;\(\text{OCR}\)相关知识。\(1\)中显示的是识别后的文本文件内容，\(2\)中显示的是识别前图片文件内容，\(3\)是相似字内容。单击区域③中的文字“天”，区域①中的“夭”字将变成“天”字；单击区域③中的文字“夭”，区域②中的“天”字并不会变化，故选项B错误；区域①中显示的是\(\text{Yunnan.txt}\)中的内容；当前正在进行识别后的修改，;\(\text{VB}\)相关知识。\(\text{frm}\)文件属于窗体文件;本每一次选择其中一张进行折叠撕开，比起上一次多了\(3\)张纸片，故需要对总数\(\text{s}\)每次循环加\(3\)，即\(\text{s}←\text{s}+3\)。;\(\text{Flash}\)软件。舞台上的“\(\text{Play}\)”按钮实例会随着该按钮元件的改变而改变。按钮有效，故测试影片，当鼠标指针经过该按钮时，指针形状会变成手形。测试影片，当鼠标指针经过该按钮时，文字“\(\text{Play}\)”消失，测试影片，当按钮按下时，按钮上呈现的文字与图中所示文字“\(\text{Play}\)”可能不同;考查\(\text{Word}\)文档的修订与审阅。观察图可知，从图1到图2所做的操作有：①删除批注  ③接受插入  ⑤接受删除  ⑦拼写检查选择忽略;\(\text{VB}\)中的算术运算符与算术表达式。若\(\text{x}\)是能被\(7\)整除的正整数则\(\text{x}\)除以\(7\)的余数为\(0\)转换为\(\text{VB}\)算术表达式为\(\text{x}\;\text{Mod}\;7=0\);信息编码。西文字符的十六进制内码不大于\(7\text{F}\)，占一个字节。中文字符占两字节。两位十六进制数表示\(1\)个字节。故可知该文本文件的内容为\(5\)个英文字符和\(2\)个中文字符，共\(9\)个字节，前\(5\)个字节为\(\text{ASCII}\)字符，最后\(4\)个字节可能为中文编码;信息的安全与网络道德规范。《全国青少年网络文明公约》倡导的行为有：要善于网上学习，不浏览不良信息；要诚实友好交流，不侮辱欺诈他人；要增强自护意识，不随意约会网友；要维护网络安全，不破坏网络秩序；要有益身心健康，不沉溺虚拟时空；;是\(\text{Excel}\)电子表格软件的应用。对单元格内容设置跨列居中，只是单元格内容显示位置发生变化，内容的单元格地址不会变化。对\(\text{A1}\)单元格的内容在\(\text{A1:E1}\)区域中跨列居中，该内容的地址还是\(\text{A1}\)，所以修改内容时还是选定\(\text{A1}\)单元格;考查选择排序。选择排序工作原理是第一次从待排序的数据元素中选出最小（或最大）的一个元素，存放在序列的起始位置，然后再从剩余的未排序元素中寻找到最小（大）元素，然后放到已排序的序列的末尾。以此类推，直到全部待排序的数据元素的个数为零。题中\(\mathrm{d(1)}\)为\(1-6\)之间的随机整数;信息的概念及其特征。信息就是对客观事物的反映，从本质上看信息是对社会、自然界的事物特征、现象、本质及规律的描述。其特征有如下：（一）依附性、（二）再生性（扩充性）、（三）可传递性、（四）可贮存性、（五）可缩性、（六）可共享性、（七）可预测性、（八）有效性和无效性、（九）可处理性。某医院的医生和护士，在为某垂危病人做手术时，通过一些医疗监视设备即时了解病人的心电图、血压等情况，从而采用不同的救治措施，最后成功挽救了病人的生命。在这个事例中，通过医疗监控设备呈现的信息所表现出来的特征，比较合理的是②共享性；③依附性和可处理性：④价值相对性；⑤时效性;了解程序设计语言程序设计语言是用于书写计算机程序的语言。语言的基础是一组记号和一组规则。根据规则由记号构成的记号串的总体就是语言。在程序设计语言中，这些记号串就是程序;信息技术（Information Technology，简称IT），是主要用于管理和处理信息所采用的各种技术的总称．一切与信息的获取，加工，表达，交流，管理和评价等有关的技术都可以称之为信息技术，古老的信息技术就可以弃之不用，有很多仍然被我们使用。;数据、信息、知识、智慧的概念。数据经过加工，可表达某种意义，则转变为信息；信息经过加工，可用于指导实践，则转变为知识；智慧是人类基于已有的知识，针对物质世界运动过程中产生的问题根据获得的信息进行分析，对比，演绎找出解决方案的能力。而本题中根据身高和体重提供的数据信息，结合Python知识编写相应的程序，从而进行算法设计程序编写来解决实际问题。故属于智慧，选项D正确。数据经过加工，可表达某种意义，则转变为信息；信息经过加工，可用于指导实践，则转变为知识；智慧是人类基于已有的知识，针对物质世界运动过程中产生的问题根据获得的信息进行分析，对比，演绎找出解决方案的能力。而本题中根据身高和体重提供的数据信息，结合Python知识编写相应的程序，从而进行算法设计程序编写来解决实际问题。故属于智慧，数据经过加工，可表达某种意义，则转变为信息；信息经过加工，可用于指导实践，则转变为知识；智慧是人类基于已有的知识，针对物质世界运动过程中产生的问题根据获得的信息进行分析，对比，演绎找出解决方案的能力。而本题中根据身高和体重提供的数据信息，结合Python知识编写相应的程序，从而进行算法设计程序编写来解决实际问题;
'''
    # for i in range(len(it_item_index_id_mapping)):
    #     if random.random() >= 0.05:
    #         continue
    #     index = it_item_index_id_mapping[i + 1]
    #     item_id = id_item_id_map[index]
    #     hint = read_file_content_as_string(index, ['hint'], True)
    #     hints = hints + hint + ";"
    # print(hints)
    score_file = []
    total_score = 0
    for i in range(len(it_item_index_id_mapping)):
        print("index: ", i + 1, " of total: ", len(it_item_index_id_mapping))
        index = it_item_index_id_mapping[i + 1]
        item_id = id_item_id_map[index]
        content = read_file_content_as_string(index, ['content'], True)
        new_hint = read_file_content_as_string(index, ['new_hint'], True)
        answer = read_file_content_as_string(index, ['answer'], True)
        request = '<回答选择题><题干>:' + content + '<提示>:' + hints
        if len(request) > 2048:
            request = request[0:2048]
        response, history = model.chat(tokenizer, request, history=[])
        # print('content: ', content)
        # print('hint: ', hint)
        # print('knowledge: ', knowledge)
        # response = response.replace(request, '')
        # response = response.replace('<题目>', '')
        # response = response.replace(content, '')
        # response = response.replace(hint, '')
        # response = response.replace(knowledge, '')
        # response = response.replace('<旧提示>', '')
        # response = response.replace('<知识点>', '')
        # response = response.replace('<生成新提示信息>', '')
        # print('new hint: ', response)
        # request = '<回答选择题><题目>:' + content + '<提示>:' + response
        # response, history = model.chat(tokenizer, request, history=[])
        # new_hint_file.append(str(id) + ',' + response)
        # if train_data_index > 500:
        #    break
        llm_original_answer = response
        extracted_answer = extract_answer_from_str(response)
        if extracted_answer is None:
            extracted_answer = response
            if len(extracted_answer) > 100:
                extracted_answer = extracted_answer[-100:]
        bot_answer = []
        if 'A' in extracted_answer:
            bot_answer.append('A')
        if 'B' in extracted_answer:
            bot_answer.append('B')
        if 'C' in extracted_answer:
            bot_answer.append('C')
        if 'D' in extracted_answer:
            bot_answer.append('D')
        print('bot answer: ', bot_answer)
        llm_answer_json = json.dumps(bot_answer)
        true_answer = []
        if 'A' in answer:
            true_answer.append('A')
        if 'B' in answer:
            true_answer.append('B')
        if 'C' in answer:
            true_answer.append('C')
        if 'D' in answer:
            true_answer.append('D')
        print('true answer: ', true_answer)
        true_answer_json = json.dumps(true_answer)

        max_length = len(bot_answer)
        if len(true_answer) > len(bot_answer):
            max_length = len(true_answer)

        match_count = 0
        for ta in true_answer:
            for ba in bot_answer:
                if ba == ta:
                    match_count += 1
                    break
        if len(true_answer) == 0:
            continue

        score = round(float(match_count) / float(len(true_answer)), 2)
        if len(bot_answer) > len(true_answer):
            score = 0.0

        print('score: ', score)

        score_file.append('1,' + str(id) + ',' + str(score))

        total_score += score
        print("current average score: " ,total_score / (i+1))
        
        save_llm_answer('ChatGLM-6B-difference-all', 
                        llm_original_answer,
                        answer, 
                        llm_answer_json, 
                        true_answer_json, 
                        score, 
                        item_id)



    # list_to_file('/home/len/kancd-data/with_new_hint/item.csv', item_file)
    # list_to_file('/home/len/kancd-data/with_new_hint/train.csv', score_file)