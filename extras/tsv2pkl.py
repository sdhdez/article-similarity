""" script/tsv_to_dict
Load tsv and save data to pickle
"""
import sys
import pickle

def load_tsv_to_dict(filename):
    """ Open tsv to dict """
    with open(filename, "r", encoding="utf-8") as tsv:
        pages = {}
        redirects = {}
        # relations = {}
        for row in tsv:
            if not row:
                break
            try:
                page_title_norm, rd_title_norm = row.strip().lower().replace('_', ' ').split("\t")
            except ValueError:
                print("Error", row, type(row))
                continue
            # print((page_title_norm, rd_title_norm))

            pages.setdefault(page_title_norm, set())
            pages.setdefault(rd_title_norm, set())
            redirects.setdefault(rd_title_norm, set())

            pages[rd_title_norm].add(rd_title_norm)
            pages[page_title_norm].add(rd_title_norm)
            redirects[rd_title_norm].add(page_title_norm)

        with open('page-redirects.pkl', 'wb') as pagered:
            pickle.dump(pages, pagered)

        with open('redirect-pages.pkl', 'wb') as redpages:
            pickle.dump(redirects, redpages)

def load_tsv_to_dict2(filename):
    """ Open tsv to dict """
    with open(filename, "r", encoding="utf-8") as tsv:
        pages = {}
        redirects = {}
        # relations = {}
        for row in tsv:
            if not row:
                break
            try:
                page_title_norm, rd_title_norm = row.strip().lower().replace('_', ' ').split("\t")
            except ValueError:
                print("Error", row, type(row))
                continue
            # print((page_title_norm, rd_title_norm))

            pages.setdefault(page_title_norm, set())
            pages.setdefault(rd_title_norm, set())
            redirects.setdefault(rd_title_norm, set())

            pages[rd_title_norm].add(rd_title_norm)
            pages[page_title_norm].add(rd_title_norm)
            redirects[rd_title_norm].add(page_title_norm)

        with open('page-redirects2.pkl', 'wb') as pagered:
            pickle.dump(pages, pagered)

        with open('redirect-pages2.pkl', 'wb') as redpages:
            pickle.dump(redirects, redpages)

def load_tsv_to_pickle(filename):
    """ Open tsv to dict """
    with open(filename, "r", encoding="utf-8") as tsv:
        equivalencies = {}
        for row in tsv:
            if not row:
                break
            try:
                page_title_norm, rd_title_norm = row.strip().lower().replace('_', ' ').split("\t")
            except ValueError:
                print("Error", row, type(row))
                continue
            if page_title_norm[-16:] == "(disambiguation)" or rd_title_norm[-16:] == "(disambiguation)":
                continue
            # print((page_title_norm, rd_title_norm))
            equivalencies.setdefault(rd_title_norm, set())
            equivalencies[rd_title_norm].add(rd_title_norm)
            equivalencies[rd_title_norm].add(page_title_norm)
            equivalencies.setdefault(page_title_norm, set())
            equivalencies[page_title_norm].add(page_title_norm)
            equivalencies[page_title_norm].add(rd_title_norm)

        with open('equivalencies.pkl', 'wb') as equiv:
            pickle.dump(equivalencies, equiv)

        print("equivalencies", len(equivalencies))

def load_pickle():
    with open('page-redirects.pkl', 'rb') as lpagered:
        pages = pickle.load(lpagered)

    with open('redirect-pages.pkl', 'rb') as lredpages:
        redirects = pickle.load(lredpages)
    return pages, redirects

def load_pickle2():
    with open('page-redirectsé.pkl', 'rb') as lpagered:
        pages = pickle.load(lpagered)

    with open('redirect-pages2.pkl', 'rb') as lredpages:
        redirects = pickle.load(lredpages)
    return pages, redirects

def load_equiv():
    with open('equivalencies.pkl', 'rb') as equiv:
        equivs = pickle.load(equiv)
    return equivs

if __name__ == "__main__":
    if sys.argv[1:]:
        # load_tsv_to_dict(sys.argv[1])
        load_tsv_to_dict2(sys.argv[1])
        # load_tsv_to_pickle(sys.argv[1])
        pass
