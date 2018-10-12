import os
from andes import system, filters
import numpy as np


class case_g:
    def __init__(self, event, t, case, save_file, e_idx):
        sys = system.PowerSystem(case)
        assert filters.guess(sys)
        assert filters.parse(sys)
        sys.setup()
        sys.pflow.run()
        sys.tds.init()
        self.sys = sys
        self.event = event
        self.t = t
        self.e_idx = e_idx
        self.case = case
        self.save_file = save_file
        self.input_check()
        self.e_count = self.t.__len__()
        self.e_string = []
        self.line_bus = sys.Line.bus1

    def input_check(self):
        if self.event.__len__() == self.t.__len__() and self.t.__len__() == self.e_idx.__len__():
            pass
        else:
            print('check input')
            os._exit(0)
        if self.event.__len__() == 0 or self.t.__len__() == 0 or self.e_idx.__len__() == 0:
            self.auto_gen()
        else:
            pass

    def auto_gen(self):
        all_event = ["GT", "LS", "LT"]
        event = []
        t = []
        e_idx = []
        for event_number in range(np.random.randint(2, 4)):
            rand_event = np.random.randint(0, 3)
            rand_t = np.random.randint(1, 11)
            event.append(all_event[rand_event])
            t.append(rand_t)
            if rand_event == 0:  # GT
                n = self.sys.Bus.idx.__len__()
                rand_bus = np.random.randint(0, n)
                e_idx.append(self.sys.Bus.idx[rand_bus])
            if rand_event == 1:  # LS
                n = self.sys.PQ.bus.__len__()
                rand_load = np.random.randint(0, n)
                e_idx.append(self.sys.PQ.bus[rand_load])
            if rand_event == 2:  # LT
                n = self.sys.Line.idx.__len__()
                rand_line = np.random.randint(0, n)
                e_idx.append(self.sys.Line.idx[rand_line])
        self.event = event
        self.t = t
        self.e_idx = e_idx

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
        file_name = '{event}_{event_place}.dm'
        e_string = g.e_string
        event = ''
        event_place = ''
        header_type = """# This case implements {event} event on {place} at t={t}s"""
        header = """# DOME format version 1.0
INCLUDE, {} \n """
        out = header.format(self.case)
        for i in range(self.e_count):
            if i == 0:
                out += header_type.format(event=self.event[i], place=self.e_idx[i], t=self.t[i])
            else:
                out += "\n"
                out += header_type.format(event=self.event[i], place=self.e_idx[i], t=self.t[i])
        out += "\n"
        for i in range(self.e_count):
            if i == 0:
                event += self.event[i]
                event_place += str(self.e_idx[i])
                out += e_string[i]
            else:
                out += "\n"
                out += e_string[i]
                event += '_'
                event_place += '_'
                event += self.event[i]
                event_place += str(self.e_idx[i])
        file_name = file_name.format(event=event, event_place=event_place)
        with open(os.path.join(save_path, file_name), 'w') as f:
            f.write(out)


# parameter specification
# TO DO auto_gen
os.chdir('C:\\Users\\zhan2\\PycharmProjects\\andes_github\\cases\\curent')
save_file = 'C:\\Users\\zhan2\\PycharmProjects\\andes_github\\demos\\Multievent\\multi_E'
case = 'WECC_WIND0.dm'
Event_list = [["GT", "LS", "LT"], ["LS", "LT"]]
t_list = [[1, 2, 3], [2, 3]]
e_idx_list = [[1, 3, "Line_0"], [3, "Line_0"]]
flag_autog = 1  # 1 indicate use random generation
case_count = 5  # define how many cases you want from random generation
if flag_autog == 1:
    for idx in range(case_count):
        g = case_g([], [], case, save_file, [])
        g.get_idx()
        g.get_e_string()
        g.make_e()
else:
    for idx in range(Event_list.__len__()):
        Event = Event_list[idx]
        t = t_list[idx]
        e_idx = e_idx_list[idx]
        g = case_g(Event, t, case, save_file, e_idx)
        g.get_idx()
        g.get_e_string()
        g.make_e()
