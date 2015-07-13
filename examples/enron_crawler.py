import os
import sys
import pickle

from email.parser import Parser
from flanker.addresslib import address
from dateutil.parser import parse as parse_datestr
from enron_utils import SUBSTITUTIONS

def scandal_predicate(headers):
    """
    See Sec 5.1 of
    http://www.kecl.ntt.co.jp/as/members/ishiguro/open/2012AISTATS.pdf
    """
    dt = parse_datestr(headers['date'])
    return dt.year == 2001 and dt.month in (8, 10)

def fix_addr_if_necessary(addr):
    return SUBSTITUTIONS.get(addr, addr)

def get_email_from_to(headers):
    sender = address.parse(fix_addr_if_necessary(headers['from']))
    if not sender:
        print "could not parse sender: {}".format(headers['from'])
        return None, None
    if sender.hostname != 'enron.com':
        # skipping b/c we only care about enron communications
        return None, None
    receivers = address.parse_list(headers['to'])
    enron_receivers = [r for r in receivers if r.hostname == 'enron.com']
    return sender, enron_receivers

def get_all_communications_in_maildir(path, predicate=None):
    print "get_all_communications_in_maildir(path={})".format(path)
    ret = []
    emails = os.listdir(path)
    for email in emails:
        p = os.path.join(path, email)
        if os.path.isdir(p):
            continue
        with open(p) as fp:
            headers = Parser().parse(fp)
            if predicate and not predicate(headers):
                continue
            #print "email={}".format(p)
            sender, receivers = get_email_from_to(headers)
            if not sender or not receivers:
                continue
            ret.append((sender, receivers))
    return ret

def get_all_communications_in_userdir(path, predicate=None):
    print "get_all_communications_in_userdir(path={})".format(path)

    # See Sec 2 of: http://www.bklimt.com/papers/2004_klimt_ecml.pdf
    blacklist = ('discussion_threads', 'notes_inbox', 'all_documents')

    def alldirs(d):
        for dirpath, dirnames, _ in os.walk(d):
            for dirname in dirnames:
                if dirname in blacklist:
                    continue
                yield os.path.join(dirpath, dirname)

    ret = []
    for folder in alldirs(path):
        ret.extend(get_all_communications_in_maildir(folder, predicate))
    return ret

def canonicalize_name(name):
    tokens = name.split('.')
    if len(tokens) != 2:
        return None
    return "{0}-{1}".format(tokens[1], tokens[0][:1])

def main():
    d, out = sys.argv[1:]
    print "analyzing dir: {}".format(d)
    print "writing results to: {}".format(out)
    names = os.listdir(d)
    communications = { name : set() for name in names }
    for name in names:
        p = os.path.join(d, name)
        fromto = get_all_communications_in_userdir(p, scandal_predicate)
        for sender, receivers in fromto:
            sender = canonicalize_name(sender.mailbox)
            if sender not in communications:
                continue
            receivers = [canonicalize_name(r.mailbox) for r in receivers]
            for receiver in receivers:
                if receiver not in communications:
                    continue
                communications[sender].add(receiver)
        print "finished with", name

    results = [(k, v) for k, v in communications.iteritems() if v]
    with open(out, "w") as fp:
        pickle.dump(results, fp)

if __name__ == '__main__':
    main()
