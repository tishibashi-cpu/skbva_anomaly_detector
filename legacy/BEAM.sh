#!/usr/bin/sh
#
#       yyyymmddhhmmss-yyyymmddhhmmss
#ttime="20160213070000-20160213220000d60"
#Etafname="20160213_Eta5.txt"
#
#echo $ttime
#
if [ $# -ne 2 ];then
 echo "INPUT ERROR !" 1>&2
 echo "number of arguments = $#" 1>&2
 echo "you need 2 arguments (yyyymmddhhmmss-yyyymmddhhmmssd60 filename)" 1>&2
 exit 1
fi
#
#
echo "time = " $1
echo "filename = " $2
#
ttime=$1
Etafname=$2
#
#
Ltemp3=$(mktemp)
Htemp0=$(mktemp)
Htemp1=$(mktemp)
Htemp2=$(mktemp)
Htemp3=$(mktemp)

# echo "Ltemp3" $Ltemp3
# echo "Htemp1" $Htemp1

function rm_tmpfile {
  [[ -f "$Ltemp3" ]] && rm -f "$Ltemp3"
  [[ -f "$Htemp0" ]] && rm -f "$Htemp0"
  [[ -f "$Htemp1" ]] && rm -f "$Htemp1"
  [[ -f "$Htemp2" ]] && rm -f "$Htemp2"
  [[ -f "$Htemp3" ]] && rm -f "$Htemp3"
}

# 正常終了したとき
trap rm_tmpfile EXIT
# 異常終了したとき
trap 'trap - EXIT; rm_tmpfile; exit -1' INT PIPE TERM
#
/usr/local/bin/kblogrd -r BMLDCCT:CURRENT,BMLDCCT:LIFE,BMHDCCT:CURRENT,BMHDCCT:LIFE -t $ttime -f kaleida BM/DCCT > $Ltemp3
#
echo "Beam current end"
#
/usr/local/bin/kblogrd -r COpLER:BEAM:LIFE,COeHER:BEAM:LIFE,CGLINJ:BKSEL:NOB_SET,CGHINJ:BKSEL:NOB_SET -t $ttime -f kaleida Misc/Base > $Htemp0
#
echo "Bucket no. end"
#
# awk 'BEGIN{FS="\t"}{$1 = ""; print}' Htemp0.txt > Htemp1.txt
# awk '$1 = $1' Htemp1.txt > Htemp2.txt
# awk '{gsub(" ", "\t"); print $0}' Htemp2.txt > Htemp3.txt
#
awk 'BEGIN{FS="\t"}{print($2, $3, $4, $5)}' $Htemp0 > $Htemp1
awk '{gsub(" ", "\t"); print $0}' $Htemp1 > $Htemp3

#
paste $Ltemp3 $Htemp3 > $Etafname
#
exit 0
#
