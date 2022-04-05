from collections import Counter
dataset = open("test_label.txt", encoding='UTF-8')
lines = []
for line in dataset.readlines():
    lines.append(line[2:])
dataset.close()
newlines = dict(Counter(lines))
ls = [key for key, value in newlines.items() if value == 1]  # 只展示重复元素
print(len(ls))
# for item in ls:
#     print(item)
# dic = {key: value for key, value in newlines.items() if value > 1}  # 展现重复元素和重复次数
# for item in dic.items():
#     print(item)

# dataset = open("test_label.txt", encoding='UTF-8')
# content = dataset.readlines()
# dataset.close()
# count = [0, 0, 0, 0]
# for line in content:
#     if line[0:2] == '0,':
#         count[0] += 1
#     elif line[0:2] == '1,':
#         count[1] += 1
#     elif line[0:2] == '2,':
#         count[2] += 1
#     elif line[0:2] == '3,':
#         count[3] += 1
# print(count)
# print(sum(count))
