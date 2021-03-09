"""
Documenter class for ANDES models.
"""

from collections import OrderedDict

from andes.utils.tab import math_wrap, make_doc_table


class Documenter:
    """
    Helper class for documenting models.

    Parameters
    ----------
    parent : Model
        The `Model` instance to document
    """

    def __init__(self, parent):
        self.parent = parent
        self.system = parent.system
        self.class_name = parent.class_name
        self.config = parent.config
        self.cache = parent.cache
        self.params = parent.params
        self.services = parent.services
        self.discrete = parent.discrete
        self.blocks = parent.blocks

    def _param_doc(self, max_width=78, export='plain'):
        """
        Export formatted model parameter documentation as a string.

        Parameters
        ----------
        max_width : int, optional = 80
            Maximum table width. If export format is ``rest`` it will be unlimited.

        export : str, optional = 'plain'
            Export format, 'plain' for plain text, 'rest' for restructuredText.

        Returns
        -------
        str
            Tabulated output in a string
        """
        if len(self.params) == 0:
            return ''

        # prepare temporary lists
        names, units, class_names = list(), list(), list()
        info, defaults, properties = list(), list(), list()
        units_rest = list()

        for p in self.params.values():
            names.append(p.name)
            class_names.append(p.class_name)
            info.append(p.info if p.info else '')
            defaults.append(p.default if p.default is not None else '')
            units.append(f'{p.unit}' if p.unit else '')
            units_rest.append(f'*{p.unit}*' if p.unit else '')

            plist = []
            for key, val in p.property.items():
                if val is True:
                    plist.append(key)
            properties.append(','.join(plist))

        # symbols based on output format
        if export == 'rest':
            symbols = [item.tex_name for item in self.params.values()]
            symbols = math_wrap(symbols, export=export)
        else:
            symbols = [item.name for item in self.params.values()]

        plain_dict = OrderedDict([('Name', names),
                                  ('Description', info),
                                  ('Default', defaults),
                                  ('Unit', units),
                                  ('Properties', properties)])

        rest_dict = OrderedDict([('Name', names),
                                 ('Symbol', symbols),
                                 ('Description', info),
                                 ('Default', defaults),
                                 ('Unit', units_rest),
                                 ('Properties', properties)])

        # convert to rows and export as table
        return make_doc_table(title='Parameters',
                              max_width=max_width,
                              export=export,
                              plain_dict=plain_dict,
                              rest_dict=rest_dict)

    def _var_doc(self, max_width=78, export='plain'):
        # variable documentation
        if len(self.cache.all_vars) == 0:
            return ''

        names, symbols, units = list(), list(), list()
        properties, info = list(), list()
        units_rest, ty = list(), list()

        for p in self.cache.all_vars.values():
            names.append(p.name)
            ty.append(p.class_name)
            info.append(p.info if p.info else '')
            units.append(p.unit if p.unit else '')
            units_rest.append(f'*{p.unit}*' if p.unit else '')

            # collect properties
            all_properties = ['v_str', 'v_setter', 'e_setter', 'v_iter']
            plist = []
            for item in all_properties:
                if (p.__dict__[item] is not None) and (p.__dict__[item] is not False):
                    plist.append(item)
            properties.append(','.join(plist))

        # replace with latex math expressions if export is ``rest``
        if export == 'rest':
            call_store = self.system.calls[self.class_name]
            symbols = math_wrap(call_store.x_latex + call_store.y_latex, export=export)

        plain_dict = OrderedDict([('Name', names),
                                  ('Type', ty),
                                  ('Description', info),
                                  ('Unit', units),
                                  ('Properties', properties)])

        rest_dict = OrderedDict([('Name', names),
                                 ('Symbol', symbols),
                                 ('Type', ty),
                                 ('Description', info),
                                 ('Unit', units_rest),
                                 ('Properties', properties)])

        return make_doc_table(title='Variables (States + Algebraics)',
                              max_width=max_width,
                              export=export,
                              plain_dict=plain_dict,
                              rest_dict=rest_dict)

    def _init_doc(self, max_width=78, export='plain'):
        """
        Variable initialization docs.
        """
        if len(self.cache.all_vars) == 0:
            return ''

        names, symbols, ivs = list(), list(), list()
        ivs_rest, ty = list(), list()

        for p in self.cache.all_vars.values():
            names.append(p.name)
            ty.append(p.class_name)
            ivs.append(p.v_str if p.v_str else '')

        # replace with latex math expressions if export is ``rest``
        if export == 'rest':
            call_store = self.system.calls[self.class_name]
            symbols = math_wrap(call_store.x_latex + call_store.y_latex, export=export)
            ivs_rest = math_wrap(call_store.init_latex.values(), export=export)

        plain_dict = OrderedDict([('Name', names),
                                  ('Type', ty),
                                  ('Initial Value', ivs),
                                  ])

        rest_dict = OrderedDict([('Name', names),
                                 ('Symbol', symbols),
                                 ('Type', ty),
                                 ('Initial Value', ivs_rest),
                                 ])

        return make_doc_table(title='Variable Initialization Equations',
                              max_width=max_width,
                              export=export,
                              plain_dict=plain_dict,
                              rest_dict=rest_dict)

    def _eq_doc(self, max_width=78, export='plain', e_code=None):
        """
        Return equation documentation.
        """
        out = ''
        if len(self.cache.all_vars) == 0:
            return out

        if e_code is None:
            e_code = ('f', 'g')
        elif isinstance(e_code, str):
            e_code = (e_code,)

        e2full = {'f': 'Differential',
                  'g': 'Algebraic'}
        e2form = {'f': "T x' = f(x, y)",
                  'g': "0 = g(x, y)"}

        e2dict = {'f': self.cache.states_and_ext,
                  'g': self.cache.algebs_and_ext}
        for e_name in e_code:
            if len(e2dict[e_name]) == 0:
                continue

            names, symbols = list(), list()
            eqs, eqs_rest = list(), list()
            lhs_names, lhs_tex_names = list(), list()
            class_names = list()

            for p in e2dict[e_name].values():
                names.append(p.name)
                class_names.append(p.class_name)
                eqs.append(p.e_str if p.e_str else '')
                if e_name == 'f':
                    lhs_names.append(p.t_const.name if p.t_const else '')
                    lhs_tex_names.append(p.t_const.tex_name if p.t_const else '')

            plain_dict = OrderedDict([('Name', names),
                                      ('Type', class_names),
                                      (f'RHS of Equation "{e2form[e_name]}"', eqs),
                                      ])

            if export == 'rest':
                call_store = self.system.calls[self.class_name]
                e2var_sym = {'f': call_store.x_latex,
                             'g': call_store.y_latex}
                e2eq_sym = {'f': call_store.f_latex,
                            'g': call_store.g_latex}

                symbols = math_wrap(e2var_sym[e_name], export=export)
                eqs_rest = math_wrap(e2eq_sym[e_name], export=export)

            rest_dict = OrderedDict([('Name', names),
                                     ('Symbol', symbols),
                                     ('Type', class_names),
                                     (f'RHS of Equation "{e2form[e_name]}"', eqs_rest),
                                     ])

            if e_name == 'f':
                plain_dict['T (LHS)'] = lhs_names
                rest_dict['T (LHS)'] = math_wrap(lhs_tex_names, export=export)

            out += make_doc_table(title=f'{e2full[e_name]} Equations',
                                  max_width=max_width,
                                  export=export,
                                  plain_dict=plain_dict,
                                  rest_dict=rest_dict)

        return out

    def _service_doc(self, max_width=78, export='plain'):
        if len(self.services) == 0:
            return ''

        names, symbols = list(), list()
        eqs, eqs_rest, class_names = list(), list(), list()

        for p in self.services.values():
            names.append(p.name)
            class_names.append(p.class_name)
            symbols.append(p.tex_name if p.tex_name is not None else '')
            eqs.append(p.v_str if p.v_str else '')

        if export == 'rest':
            call_store = self.system.calls[self.class_name]
            symbols = math_wrap(symbols, export=export)
            eqs_rest = math_wrap(call_store.s_latex, export=export)

        plain_dict = OrderedDict([('Name', names),
                                  ('Equation', eqs),
                                  ('Type', class_names)])

        rest_dict = OrderedDict([('Name', names),
                                 ('Symbol', symbols),
                                 ('Equation', eqs_rest),
                                 ('Type', class_names)])

        return make_doc_table(title='Services',
                              max_width=max_width,
                              export=export,
                              plain_dict=plain_dict,
                              rest_dict=rest_dict)

    def _discrete_doc(self, max_width=78, export='plain'):
        if len(self.discrete) == 0:
            return ''

        names, symbols, info = list(), list(), list()
        class_names = list()

        for p in self.discrete.values():
            names.append(p.name)
            class_names.append(p.class_name)
            info.append(p.info if p.info else '')

        if export == 'rest':
            symbols = math_wrap([item.tex_name for item in self.discrete.values()], export=export)
        plain_dict = OrderedDict([('Name', names),
                                  ('Type', class_names),
                                  ('Info', info)])

        rest_dict = OrderedDict([('Name', names),
                                 ('Symbol', symbols),
                                 ('Type', class_names),
                                 ('Info', info)])

        return make_doc_table(title='Discrete',
                              max_width=max_width,
                              export=export,
                              plain_dict=plain_dict,
                              rest_dict=rest_dict)

    def _block_doc(self, max_width=78, export='plain'):
        """
        Documentation for blocks.
        """
        if len(self.blocks) == 0:
            return ''

        names, symbols, info = list(), list(), list()
        class_names = list()

        for p in self.blocks.values():
            names.append(p.name)
            class_names.append(p.class_name)
            info.append(p.info if p.info else '')

        if export == 'rest':
            symbols = math_wrap([item.tex_name for item in self.blocks.values()], export=export)

        plain_dict = OrderedDict([('Name', names),
                                  ('Type', class_names),
                                  ('Info', info)])

        rest_dict = OrderedDict([('Name', names),
                                 ('Symbol', symbols),
                                 ('Type', class_names),
                                 ('Info', info)])

        return make_doc_table(title='Blocks',
                              max_width=max_width,
                              export=export,
                              plain_dict=plain_dict,
                              rest_dict=rest_dict)

    def get(self, max_width=78, export='plain'):
        """
        Return the model documentation in table-formatted string.

        Parameters
        ----------
        max_width : int
            Maximum table width. Automatically et to 0 if format is ``rest``.
        export : str, ('plain', 'rest')
            Export format. Use fancy table if is ``rest``.

        Returns
        -------
        str
            A string with the documentations.
        """
        out = ''
        if export == 'rest':
            max_width = 0
            model_header = '-' * 80 + '\n'
            out += f'.. _{self.class_name}:\n\n'
        else:
            model_header = ''

        if export == 'rest':
            out += model_header + f'{self.class_name}\n' + model_header
            out += f'\nGroup {self.parent.group}_\n\n'
        else:
            out += model_header + f'Model <{self.class_name}> in Group <{self.parent.group}>\n' + model_header

        if self.__doc__ is not None:
            if self.parent.__doc__ is not None:
                out += self.parent.__doc__
            out += '\n'  # this fixes the indentation for the next line

        # add tables
        out += self._param_doc(max_width=max_width, export=export)
        out += self._var_doc(max_width=max_width, export=export)
        out += self._init_doc(max_width=max_width, export=export)
        out += self._eq_doc(max_width=max_width, export=export)
        out += self._service_doc(max_width=max_width, export=export)
        out += self._discrete_doc(max_width=max_width, export=export)
        out += self._block_doc(max_width=max_width, export=export)
        out += self.config.doc(max_width=max_width, export=export)

        return out
