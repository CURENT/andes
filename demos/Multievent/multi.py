import os
from andes import system, filters


class case_g:
    def __init__(self, event, t, case, save_file, e_idx):
        if event.__len__() == t.__len__() and t.__len__() == e_idx.__len__():
            pass
        else:
            print('check input')
            os._exit(0)
        sys = system.PowerSystem(case)
        assert filters.guess(sys)
        assert filters.parse(sys)
        sys.setup()
        sys.pflow.run()
        sys.tds.init()

        self.sys = sys
        self.event = event
        self.t = t
        self.case = case
        self.save_file = save_file
        self.e_count = t.__len__()
        self.e_idx = e_idx
        self.e_string = []
        self.line_bus = sys.Line.bus1

    def get_idx(self):
        self.wecc_bus = self.sys.Bus.idx
        self.wecc_gen = self.sys.Syn6a.on_bus(self.wecc_bus)
        self.wecc_load = self.sys.PQ.on_bus(self.wecc_bus)
        self.wecc_line = self.sys.Line.link_bus(self.wecc_bus)

    def e_switch(self, idx):
        type = self.event[idx]
        if type == "GT":
            switcher = 'GenTrip, t1 = {}, gen = {}'
            e_string = switcher.format(self.t[idx], self.e_idx[idx])
        elif type == "LS":
            switcher = 'LoadShed, group=\"StaticLoad\", t1 = {}, load = \"{}\"'
            b_idx = self.wecc_bus.index(self.e_idx[idx])
            load = self.wecc_load[b_idx]
            e_string = switcher.format(self.t[idx], load)
        elif type == "LT":
            switcher = 'Breaker, t1 = {}, bus={bus}, u1=1 , line = \"{line}\"'
            bus1_idx = self.sys.Line.idx.index(self.e_idx[idx])
            bus = self.line_bus[bus1_idx]
            e_string = switcher.format(self.t[idx], bus=bus, line=self.e_idx[idx])
        else:
            print('event not included')
            os._exit(0)
        return e_string

    def get_e_string(self):
        for idx in range(self.e_count):
            e_string = self.e_switch(idx)
            self.e_string.append(e_string)

    def make_e(self):
        save_path = self.save_file
        # To Do: doc string, file name
        file_name = '{event}_{bus}.dm'
        e_string = g.e_string
        event1 = ''
        bus1 = ''
        # element = ''
        header = """# DOME format version 1.0
        INCLUDE, {}
        """
        out = header.format(self.case)
        for i in range(self.e_count):
            if i == 0:
                out += e_string[i]
                event1 += self.event[i]
                bus1 += str(self.e_idx[i])
            else:
                out += "\n"
                out += e_string[i]
                event1 += self.event[i]
                bus1 += str(self.e_idx[i])
                file_name = file_name.format(event=event1, bus=bus1)
        with open(os.path.join(save_path, file_name), 'w') as f:
            f.write(out)


# parameter specification
# TO DO g.auto_gen
os.chdir('C:\\Users\\zhan2\\PycharmProjects\\andes_github\\cases\\curent')
save_file = 'C:\\Users\\zhan2\\PycharmProjects\\andes_github\\demos\\Multievent\\multi_E'
case = 'WECC_WIND0.dm'
Event_list = [["GT", "LS", "LT"], ["LS", "LT"]]
t_list = [[1, 2, 3], [2, 3]]
e_idx_list = [[1, 3, "Line_0"], [3, "Line_0"]]


for idx in range(Event_list.__len__()):
    Event = Event_list[idx]
    t = t_list[idx]
    e_idx = e_idx_list[idx]
    g = case_g(Event, t, case, save_file, e_idx)
    g.get_idx()
    g.get_e_string()
    g.make_e()
