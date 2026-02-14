"""
DAE compaction helpers for System.
"""

#  [ANDES] (C)2015-2024 Hantao Cui
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 3 of the License, or
#  (at your option) any later version.

import logging

from andes.shared import np

logger = logging.getLogger(__name__)


class DAECompactor:
    """
    Handle DAE algebraic-variable compaction for replaced static devices.
    """

    def __init__(self, system):
        self.system = system

    @staticmethod
    def _break_var_contiguity(var):
        """Mark a variable as non-contiguous so ``set_var_arrays`` allocates standalone arrays."""
        var._contiguous = False
        var.v_inplace = False
        var.e_inplace = False

    def compact_dae(self):
        """
        Detect replaced static devices and compact ``dae.y`` by removing
        their algebraic variable slots.

        Called during TDS initialization after ``init()``.
        """
        excluded = self._detect_replaced_devices()
        if not excluded:
            return

        old_to_new, new_m, need_sink, all_replaced_models = \
            self._build_y_compaction_map(excluded)

        self._compact_dae_y(old_to_new, new_m, need_sink,
                            all_replaced_models)

    def _detect_replaced_devices(self):
        """
        Scan TDS models for ``IdxParam`` with ``replaces=True``, mark
        referenced static devices as ``_replaced``, and set ``u=0``.

        Returns
        -------
        dict
            ``{model_name: set_of_replaced_uids}``
        """
        system = self.system
        excluded = {}

        for mdl in system.exist.tds.values():
            if mdl.n == 0:
                continue
            for p_instance in mdl.idx_params.values():
                if not p_instance.replaces:
                    continue

                group = system.groups[p_instance.model]

                for i in range(mdl.n):
                    # only consider online dynamic devices as replacements
                    if hasattr(mdl, 'u') and mdl.u.v[i] != 1:
                        continue
                    target_idx = p_instance.v[i]
                    if target_idx is None:
                        continue

                    target_model = group.idx2model(target_idx)
                    target_uid = target_model.idx2uid(target_idx)
                    model_name = target_model.class_name

                    if target_model._replaced is None:
                        target_model._replaced = np.zeros(target_model.n, dtype=bool)
                    target_model._replaced[target_uid] = True

                    excluded.setdefault(model_name, set()).add(target_uid)

        # set u=0 and ue=0 on all replaced devices
        for model_name, uids in excluded.items():
            mdl = system.__dict__[model_name]
            for uid in uids:
                mdl.u.v[uid] = 0
                mdl.ue.v[uid] = 0

        return excluded

    def _build_y_compaction_map(self, excluded):
        """
        Build mapping from old ``dae.y`` indices to new compact indices.

        Parameters
        ----------
        excluded : dict
            ``{model_name: set_of_replaced_uids}`` from ``_detect_replaced_devices``

        Returns
        -------
        old_to_new : np.ndarray
            Mapping array (``-1`` for removed indices).
        new_m : int
            New algebraic variable count (excluding sink).
        need_sink : bool
            Whether a sink slot is needed for partial replacement.
        all_replaced_models : set
            Model names where every device is replaced.
        """
        system = self.system
        old_m = system.dae.m
        remove_indices = set()
        all_replaced_models = set()

        for model_name, uids in excluded.items():
            mdl = system.__dict__[model_name]

            if len(uids) == mdl.n:
                all_replaced_models.add(model_name)

            # collect internal Algeb addresses for replaced devices
            for var in mdl.algebs.values():
                for uid in uids:
                    remove_indices.add(var.a[uid])

        # build old-to-new mapping
        old_to_new = np.full(old_m, -1, dtype=int)
        new_idx = 0
        for old_idx in range(old_m):
            if old_idx not in remove_indices:
                old_to_new[old_idx] = new_idx
                new_idx += 1
        new_m = new_idx

        # need sink if any model has partial replacement
        need_sink = len(all_replaced_models) < len(excluded)

        return old_to_new, new_m, need_sink, all_replaced_models

    def _compact_dae_y(self, old_to_new, new_m, need_sink,
                       all_replaced_models):
        """
        Compact ``dae.y`` by removing replaced devices' algebraic variable
        slots, remap addresses, and refresh model bindings.

        Parameters
        ----------
        old_to_new : np.ndarray
            Mapping from old to new indices (``-1`` for removed).
        new_m : int
            New algebraic variable count (excluding sink).
        need_sink : bool
            Whether a sink slot is needed.
        all_replaced_models : set
            Models where every device is replaced.
        """
        system = self.system
        old_m = system.dae.m
        keep = old_to_new >= 0

        # --- resize / compact arrays ---

        if need_sink:
            sink_idx = new_m
            system.dae.m = new_m + 1
        else:
            sink_idx = None
            system.dae.m = new_m

        system._y_sink_idx = sink_idx
        system._all_replaced_models = all_replaced_models

        new_y = np.zeros(system.dae.m)
        new_y[old_to_new[keep]] = system.dae.y[keep]
        system.dae.y = new_y
        system.dae.g = np.zeros(system.dae.m)

        old_y_name = system.dae.y_name[:]
        old_y_tex_name = system.dae.y_tex_name[:]
        system.dae.y_name = [''] * system.dae.m
        system.dae.y_tex_name = [''] * system.dae.m
        old_y_map = dict(system.dae.y_map)
        system.dae.y_map = {}
        for old_idx in np.where(keep)[0]:
            ni = old_to_new[old_idx]
            system.dae.y_name[ni] = old_y_name[old_idx]
            system.dae.y_tex_name[ni] = old_y_tex_name[old_idx]
            if old_idx in old_y_map:
                system.dae.y_map[ni] = old_y_map[old_idx]

        # --- remap var.a and break contiguity ---

        def remap_a(a):
            new_a = old_to_new[a]
            if need_sink:
                new_a[new_a == -1] = sink_idx
            return new_a

        for mdl in system.exist.pflow_tds.values():
            if mdl.n == 0 or mdl.class_name in all_replaced_models:
                continue
            for var in mdl.cache.algebs_and_ext.values():
                var.a = remap_a(var.a)
                self._break_var_contiguity(var)

        # break contiguity for all-replaced models so set_var_arrays
        # allocates fresh arrays instead of in-place views into dae.y
        for model_name in all_replaced_models:
            mdl = system.__dict__[model_name]
            for var in mdl.cache.vars_int.values():
                self._break_var_contiguity(var)

        # --- rebind views and refresh model inputs ---

        system.set_var_arrays(models=system.exist.pflow_tds)
        for model in system.exist.pflow_tds.values():
            if model.n > 0:
                model.get_inputs(refresh=True)

        # --- log ---

        n_removed = np.count_nonzero(old_to_new == -1)
        logger.info("DAE compaction: removed %d algebraic variable slots "
                    "(m: %d -> %d)", n_removed, old_m, system.dae.m)
        if need_sink:
            logger.debug("  Sink slot at index %d", sink_idx)
        if all_replaced_models:
            logger.debug("  All-replaced models: %s",
                         ', '.join(sorted(all_replaced_models)))
