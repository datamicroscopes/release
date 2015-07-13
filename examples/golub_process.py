import sys

def is_integer(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def main():
    fname = sys.argv[1]
    with open(fname) as fp:
        lines = [l.strip() for l in fp.readlines()]
        header = lines[0]
        lines = lines[1:]
    header = header.split("\t")
    lines = [l.split("\t") for l in lines if l]

    def column(lines, col):
        return [l[col] for l in lines]

    gene_idents = column(lines, 1)
    columns = [idx for idx, cname in enumerate(header) if is_integer(cname)]
    relation = [[int(l[c]) for c in columns] for l in lines]

    with open("golub_utils.py", "w") as fp:
        print >>fp, "GENE_IDENTS =", gene_idents
        print >>fp, "GENE_DATA =", relation

if __name__ == '__main__':
    main()
