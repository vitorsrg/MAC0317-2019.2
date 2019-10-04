#!/usr/bin/env bash

set -ev

#region activate
if [ ! -d './.env' ]; then
  virtualenv ./.env -p $(which python3.7)
fi

source ./.env/bin/activate
export PATH=$(pwd)/.env/bin:$PATH

pip install argh mido numpy scipy
#endregion

#region generate
waitforjobs() {
    if [[ $(ps -af | grep $$ | wc -l) -ge $(($1 - 3)) ]]; then 
        wait -n
    fi
}


mkdir -p ./wav

for adsrfile in ./adsr/*; do
    for partfile in ./part/*; do
        wavfile="./wav/$(basename -- "$partfile" .part).$(basename -- "$adsrfile" .adsr).44100.wav"
        python synthesizer.py part2wav "$adsrfile" "$partfile" "$wavfile" --samplerate 44100 &
        waitforjobs $(nproc --all)
    done
    for midifile in ./mid/*; do
        wavfile="./wav/$(basename -- "$midifile" .mid).$(basename -- "$adsrfile" .adsr).44100.wav"
        python synthesizer.py midi2wav "$adsrfile" "$midifile" "$wavfile" --samplerate 44100 &
        waitforjobs $(nproc --all)
    done
done

for adsrfile in ./adsr/piano.adsr; do
    for partfile in ./part/A_harmonic.part ./part/C_octave.part; do
        for samplerate in 44100 22050 11025 5512 2556; do
            wavfile="./wav/$(basename -- "$partfile" .part).$(basename -- "$adsrfile" .adsr).$samplerate.wav"
            python synthesizer.py part2wav "$adsrfile" "$partfile" "$wavfile" --samplerate $samplerate &
            waitforjobs $(nproc --all)
        done
    done
done

wait
#endregion

tar -cvzf 10258862.tar.gz \
    ./adsr ./img ./mid ./part \
    ./README.md ./synthesizer.py ./test.sh
