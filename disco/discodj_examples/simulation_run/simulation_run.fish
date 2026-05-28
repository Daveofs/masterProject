# Fish completions for simulation_run

# -----------------------------
# Help option
# -----------------------------
complete -c simulation_run -s h -l help \
         -d 'show this help message and exit'

# -----------------------------
# Options that take arbitrary arguments (never complete files)
# -----------------------------
for opt in name boxsize res res-pm n-order a-ini a-end cosmo precision \
           numsteps num-chunks stepper time-var grad-kernel-order worder \
           gr-correction-res \
           n-buffer-part-factor n-buffer-part-fraction \
           n-buffer-grid-factor n-buffer-grid-fraction \
           lightcone-size-factor lightcone-size-fraction \
           healpix-nside \
           fof-link-length fof-pad-factor fof-alloc-fac-nodes fof-alloc-fac-ilist  \
           fof-alloc-fac-distr-links fof-npart-min  \
           timeout-for-first-sync-point  \
           slice-axis seed-ngenic z-ini
    complete -c simulation_run -l $opt -r \
             -n "not __fish_seen_argument -l $opt" \
             -d "set $opt"

    complete -c simulation_run \
             -n "__fish_seen_argument -l $opt" \
             -f \
             -d "argument for $opt"
end

complete -c simulation_run -l run-mode -r \
         -n "not __fish_seen_argument -l run-mode" \
         -d "set run mode"
complete -c simulation_run \
         -n "__fish_seen_argument -l run-mode" \
         -a "cpu gpu gpu-one-process-per-node singlegpu singlenode singlecpu" \
         -d "run mode value"

complete -c simulation_run -l linear-ps-file -r \
         -n "not __fish_seen_argument -l linear-ps-file" \
         -d "set linear_ps_file"
complete -c simulation_run \
         -n "__fish_seen_argument -l linear-ps-file" \
         -a '(__fish_complete_suffix .asdf)' \
         -d "linear power spectrum file"

complete -c simulation_run -l gr-correction-file -r \
         -n "not __fish_seen_argument -l gr-correction-file" \
         -d "set gr_correction_file"
complete -c simulation_run \
         -n "__fish_seen_argument -l gr-correction-file" \
         -a '(__fish_complete_suffix .asdf)' \
         -d "gr correction file"

for flag in single-device no-single-device \
            padded-sim no-padded-sim \
            gr-correction-postprocessing no-gr-correction-postprocessing \
            gr-correction-lightcone no-gr-correction-lightcone \
            double-precision-timetables no-double-precision-timetables \
            healpix no-healpix \
            lightcone no-lightcone \
            rsd no-rsd \
            use-distributed-scatter-gather no-use-distributed-scatter-gather \
            use-vjp-scatter no-use-vjp-scatter \
            use-vjp-gather no-use-vjp-gather \
            scatter-gather-check no-scatter-gather-check \
            fNL-test no-fNL-test \
            use-fof-callback no-use-fof-callback \
            calculate-final-density-field no-calculate-final-density-field \
            calculate-fof no-calculate-fof \
            fof-stats no-fof-stats \
            dump-xla no-dump-xla \
            save-final-field no-save-final-field \
            save-hdf5-snapshot no-save-hdf5-snapshot \
            quiet-compile no-quiet-compile \
            powerspectrum no-powerspectrum \
            double export-config
    complete -c simulation_run -l $flag \
             -n "not __fish_seen_argument -l $flag" \
             -d "toggle $flag"
end

complete -c simulation_run -n '__fish_is_first_arg' \
         -a '(__fish_complete_suffix .toml)' \
         -d 'path to config file'
