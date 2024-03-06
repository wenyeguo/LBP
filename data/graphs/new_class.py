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

    def set_domain(self, domain):
        self.domain = domain

    def set_substrings(self, substring):
        self.substrings = substring

    def same_list(self, x, y):
        if len(x) == len(y):
            for val in x:
                if val not in y:
                    return False
            return True
        else:
            return False


    def isSame(self, obj):
        substrings = obj.get_substrings()
        domain = obj.get_domain()
        ip = obj.get_address()
        nameserver = obj.get_nameserver()
        label = obj.get_label()
        if self.domain == domain:
            if self.same_list(self.IP, ip):
                if self.same_list(self.nameserver, nameserver):
                    if self.same_list(self.substrings, substrings):
                        return True
        if self.label != label:
            print(self.url, self.label)
        return False

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
