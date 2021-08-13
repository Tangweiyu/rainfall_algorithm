import xlrd
import csv
import codecs
# 将excel文件转换为csv文件便于pandas库的使用
def xlsx_to_csv():
    workbook = xlrd.open_workbook('Result0526/new_2012-2020.xlsx')
    table = workbook.sheet_by_index(0)
    with codecs.open('new_data.csv', 'w', encoding='utf-8') as f:
        write = csv.writer(f)
        for row_num in range(table.nrows):
            row_value = table.row_values(row_num)
            write.writerow(row_value)
xlsx_to_csv()