import part1 as p1
import part2 as p2
import part3 as p3
import part4 as p4
import part5 as p5
import part6 as p6

state = 'start'

while state != 'exit':
    if state == 'start':
        print("Select part 1 to 6 to run (enter only the number [1, 2, 3, 4, 5, 6])\nEnter 0 to exit:")
        part = input()
        state = 'part'+part

    elif state == 'part1':
        print("Enter the desired number of pages to scrap\nIf you want to run with default settings just hit enter (default 2000):")
        try:
            n = int(input())
        except:
            n = 2000
        print(
            "Enter the starting_urls of SemanticScholar\nIf you want to run with default settings just hit enter:")
        ignore_urls = False
        try:
            start_urls = list(input())
        except:
            start_urls = None
        if len(start_urls) < 1:
            start_urls = None
        p1.scrap(desired_number_of_papers=n, start_urls=start_urls)
        print("The result will be saved in scholar.json")
        state = 'start'

    elif state == 'part2':
        print("Enter the host_ip of Elasticsearch server (localhost for example):")
        host_ip = input()
        print("Enter the port_number of Elasticsearch server (9200 for example):")
        port = int(input())
        print("Do you want to make index or delete it? (name of index is always paper_index)\nm: make index, c:clear index")
        com = input()
        if com == 'm':
            print("Input JSON dir (./scholar.json for example):")
            json_dir = input()
            p2.make_index(host_ip=host_ip, port_number=port, json_dir=json_dir)
        elif com == 'c':
            p2.clear_index(host_ip=host_ip, port_number=port)
            # p2.clear_index_docs(host_ip=host_ip, port_number=port)
        state = 'start'

    elif state == 'part3':
        print("Enter the host_ip of Elasticsearch server (localhost for example):")
        host_ip = input()
        print("Enter the port_number of Elasticsearch server (9200 for example):")
        port = int(input())
        print("Enter alpha of pagerank algm:")
        alpha = float(input())
        p3.set_pagerank_in_es(host_ip=host_ip, port_number=port, alpha=alpha)
        state = 'start'

    elif state == 'part4':
        print("Enter the host_ip of Elasticsearch server (localhost for example):")
        host_ip = input()
        print("Enter the port_number of Elasticsearch server (9200 for example):")
        port = int(input())
        print("Enter query:")
        q = input()
        print("Enter date:")
        d = int(input())
        print("Enter weight of title, abstract, date respectively (space separated")
        weights = list(map(float, input().split()))
        print("Pagerank involved? y/n?")
        pg = input()
        if pg == 'y':
            pg = True
        else:
            pg = False
        print("Enter size of returned search results")
        s = int(input())
        terms = [q, q, str(d)]
        p4.search(terms=terms, weights=weights, host_ip=host_ip, port_number=port, use_page_rank=pg, size=s)
        state = 'start'

    elif state == 'part5':
        print("Enter the host_ip of Elasticsearch server (localhost for example):")
        host_ip = input()
        print("Enter the port_number of Elasticsearch server (9200 for example):")
        port = int(input())
        print("Enter number of authors required:")
        n = int(input())
        p5.hits(required_number_of_authors=n, host_ip=host_ip, port_number=port)
        state = 'start'

    elif state == 'part6':
        print("Enter datapath:")
        dp = input()
        p6.find_c(data_path=dp)
        state = 'start'

    elif state == 'part0':
        state = 'exit'
