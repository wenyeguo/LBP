import pickle


class URL:
    def __init__(self, url, domain, substrings, IP, nameserver, label):
        self.url = url
        self.domain = domain
        self.substrings = substrings
        self.IP = IP
        self.nameserver = nameserver
        self.label = label


    def get_url(self):
        return self.url

    def get_domain(self):
        return self.domain

    def get_substrings(self):
        return self.substrings

    def get_address(self):
        return self.IP

    def get_nameserver(self):
        return self.nameserver

    def get_label(self):
        return self.label

    def set_address(self, address):
        self.IP = address

    def set_nameserver(self, nameserver):
        self.nameserver = nameserver

    def set_substrings(self, substring):
        self.substrings = substring

    def print_self(self):
        print("URL:", self.get_url())
        print("Domain:", self.get_domain())
        print("Substrings:", self.get_substrings())
        print("IP:", self.get_address())
        print("Nameserver:", self.get_nameserver())
        print("Label:", self.get_label())


def store_data(data, filename):
    file = open(filename, 'wb')
    pickle.dump(data, file)
    file.close()


def load_data(filename):
    file = open(filename, "rb")
    data = pickle.load(file)
    file.close()
    return data


def unique(l):
    list_set = set(l)
    unique_list = (list(list_set))
    return unique_list
