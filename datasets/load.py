import json


def main():
    subset_names = ['computer_science', 'medicine', 'biology']
    for subset_name in subset_names:
        data = json.load(open(f'datasets/{subset_name}_.json'))
        for item in data:
            print(item.keys())  
            # print(item['article_title'] + '\n')
            print(item['article_content'] + '\n')
            # print(item['paper_abstract'] + '\n')
            # print(item['paper_content'])
            input()


if __name__ == '__main__':
    main()
