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
Dtemp0=$(mktemp)
Dtemp1=$(mktemp)
Dtemp2=$(mktemp)
Dtemp3=$(mktemp)
Btemp0=$(mktemp)
Btemp1=$(mktemp)
Btemp2=$(mktemp)
Btemp3=$(mktemp)

# echo "Ltemp3" $Ltemp3
# echo "Htemp1" $Htemp1
# echo "Btemp2" $Btemp2

function rm_tmpfile {
  [[ -f "$Ltemp3" ]] && rm -f "$Ltemp3"
  [[ -f "$Htemp0" ]] && rm -f "$Htemp0"
  [[ -f "$Htemp1" ]] && rm -f "$Htemp1"
  [[ -f "$Htemp2" ]] && rm -f "$Htemp2"
  [[ -f "$Htemp3" ]] && rm -f "$Htemp3"
  [[ -f "$Dtemp0" ]] && rm -f "$Dtemp0"
  [[ -f "$Dtemp1" ]] && rm -f "$Dtemp1"
  [[ -f "$Dtemp2" ]] && rm -f "$Dtemp2"
  [[ -f "$Dtemp3" ]] && rm -f "$Dtemp3"
  [[ -f "$Btemp0" ]] && rm -f "$Btemp0"
  [[ -f "$Btemp1" ]] && rm -f "$Btemp1"
  [[ -f "$Btemp2" ]] && rm -f "$Btemp2"
  [[ -f "$Btemp3" ]] && rm -f "$Btemp3"
}

#         
trap rm_tmpfile EXIT
#         
trap 'trap - EXIT; rm_tmpfile; exit -1' INT PIPE TERM
#
/usr/local/bin/kblogrd -r VAHCCG:D04_H02:PRES,VAHCCG:D04_H03:PRES,VAHCCG:D04_H04:PRES,VAHCCG:D04_H05:PRES,VAHCCG:D04_H06:PRES,VAHCCG:D04_H06A:PRES,VAHCCG:D04_H07:PRES,VAHCCG:D04_H08:PRES,VAHCCG:D04_H08X:PRES,VAHCCG:D04_H09:PRES,VAHCCG:D04_H10:PRES,VAHCCG:D04_H11:PRES,VAHCCG:D04_H12:PRES,VAHCCG:D04_H13:PRES -t $ttime -f kaleida VA/CCG > $Ltemp3
#
echo "1 end"
#
/usr/local/bin/kblogrd -r VAHCCG:D04_H14:PRES,VAHCCG:D04_H15:PRES,VAHCCG:D04_H16:PRES,VAHCCG:D04_H17:PRES,VAHCCG:D04_H18:PRES,VAHCCG:D04_H19:PRES,VAHCCG:D04_H20:PRES,VAHCCG:D04_H21:PRES,VAHCCG:D04_H22:PRES,VAHCCG:D04_H23:PRES,VAHCCG:D04_H0X:PRES -t $ttime -f kaleida VA/CCG > $Htemp0
#
echo "2 end"
#
awk 'BEGIN{FS="\t"}{$1 = ""; print}' $Htemp0 > $Htemp1
awk '$1 = $1' $Htemp1 > $Htemp2
awk '{gsub(" ", "\t"); print $0}' $Htemp2 > $Htemp3
#
/usr/local/bin/kblogrd -r BMLDCCT:CURRENT,BMLDCCT:LIFE,BMHDCCT:CURRENT,BMHDCCT:LIFE -t $ttime -f kaleida BM/DCCT > $Dtemp0
#
echo "Current end"
#
awk 'BEGIN{FS="\t"}{print($2, $3, $4, $5)}' $Dtemp0 > $Dtemp1
awk '{gsub(" ", "\t"); print $0}' $Dtemp1 > $Dtemp3

#
/usr/local/bin/kblogrd -r COpLER:BEAM:LIFE,COeHER:BEAM:LIFE,CGHINJ:BKSEL:NOB_SET -t $ttime -f kaleida Misc/Base > $Btemp0
#
echo "Life end"
#
awk 'BEGIN{FS="\t"}{print($2, $3, $4)}' $Btemp0 > $Btemp1
awk '{gsub(" ", "\t"); print $0}' $Btemp1 > $Btemp3
#
paste $Ltemp3 $Htemp3 $Dtemp3 $Btemp3 > $Etafname
#
exit 0
#
