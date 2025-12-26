#!/usr/local/bin/python3

### This script takes phaser summary file and ManualCuration pred_tab file
### and matches those with each other to give a summary of overlap

import sys,os,time,datetime
from operator import itemgetter

phasis  = str(sys.argv[1])
ManualCuration = str(sys.argv[2])
#revferno = str(sys.argv[3])
flanksize= 300              ## Flaking region used to match at head and tail

def phasis_parse(phasis):
    '''
    parses phasis file
    '''
    print("\nFn: phasis_parse ####################")
    phasisL = []
    phasisD = {}

    fh_in   = open(phasis,'r')
    fh_in.readline()
    afile   = fh_in.readlines()
    fh_in.close()

    acount = 0
    for i in afile:
        ent     = i.strip('\n').split('\t')
        aname   = ent[0]
        apval   = ent[1]
        achr    = ent[2]
        astart  = ent[3]
        aend    = ent[4]
        library   = ent[5]
        # print(xtrig)
        #if xtrig    == "NONE":
        #    library   = "na"
        #    aindex  = "na"
        #    aloci   = "na"
        #else:
            #library   = xtrig[0]
            #aindex  = xtrig[3]
            #aloci   = xtrig[2]
        #    pass
        avalue  = (aname,achr,astart,aend,apval,library)
        # print("phasis parsed:",avalue)
        phasisL.append(avalue)
        phasisD[aname] = avalue
        acount          +=1

    print("PHAS in file:%s| PHAS cached:%s" % (acount,len(phasisL)))

    return phasisL,phasisD

def ManualCuration_parse(ManualCuration):
    '''
    parses phasetank file
    '''

    print("\nFn: ManualCuration_parse ###################")
    ManualCurationL = []
    ManualCurationD = {}

    fh_in   = open(ManualCuration,'r')
    fh_in.readline()
    afile   = fh_in.readlines()
    fh_in.close()

    acount = 0
    for i in afile:
        # print(i)
        ent     = i.strip('\n').split('\t')
        aname   = ent[0]
        ascore  = ent[4]
        achr    = ent[1]
        astart  = ent[2]
        aend  = ent[3]
        library   = ent[5]
        #if xtrig != "NONE":
            #library = xtrig.split('-')[1]
        #    pass
        #else:
        #    library = xtrig

        avalue  = (aname,achr,astart,aend,ascore,library)
        # print("PhaseTank parsed:",avalue)
        ManualCurationL.append(avalue)
        ManualCurationD[aname] = avalue
        acount          +=1

    print("PHAS in file:%s| PHAS cached:%s" % (acount,len(ManualCurationL)))

    return ManualCurationL,ManualCurationD

def matchPHAS(phasisL, ManualCurationL):
    '''
    Matches results and generates summary
    '''

    print("\nFn: matchPHAS ######################")

    # Output
    outfile     = "matched_%s.txt" % (datetime.datetime.now().strftime("%m_%d_%H_%M"))
    summaryfile = "summary_%s.txt" % (datetime.datetime.now().strftime("%m_%d_%H_%M"))
    fh_out      = open(outfile,'w')
    fh_out.write("phasis-name\tphasis-chr\tphasis-start\tphasis-end\tphasis-pval\tlibrary\tManualCuration-name\tManualCuration-chr\tManualCuration-start\tManualCuration-end\tManualCuration-score\tManualCuration-trig\tmatch-side\n")
    sm_out      = open(summaryfile,'w')

    # Store
    amatched    = 0   # phasis counter for matched
    bmatched    = 0   # ManualCuration counter for matched
    aset        = set()  # Store unique in phasis
    bset        = set()  # Store unique in ManualCuration
    matchedSet  = set()  # Store matched in both

    for aent in phasisL:
        matchflag = False
        aname, achr, astart, aend, apval, library = aent
        astart, aend = int(astart) - flanksize, int(aend) + flanksize
        for bent in ManualCurationL:
            bname, bchr, bstart, bend, bpval, btrig = bent
            bstart, bend = int(bstart), int(bend)
            if achr == bchr:                
                # Check for any overlap
                if astart <= bend and bstart <= aend:
                    # Match found
                    matchflag   = True
                    amatched    +=1
                    bmatched    +=1
                    matchedSet.add(aname)
                    matchedSet.add(bname)

                    # Determine which side overlaps
                    if bstart <= astart <= bend or bstart <= aend <= bend:
                        overlapside = 5 if bstart <= astart else 3
                    else:
                        overlapside = 0  # Contained

                    fh_out.write(f"{aname}\t{achr}\t{astart}\t{aend}\t{apval}\t{library}\t{bname}\t{bchr}\t{bstart}\t{bend}\t{bpval}\t{btrig}\t{overlapside}\n")

        # Unmatched phasis entry
        if not matchflag:
            fh_out.write(f"{aname}\t{achr}\t{astart}\t{aend}\t{apval}\t{library}\n")
            aset.add(aname)

    # Write unmatched ManualCuration entries
    for bent in ManualCurationL:
        bname, bchr, bstart, bend, bpval, btrig = bent
        if bname not in matchedSet:
            fh_out.write(f"x\tx\tx\tx\tx\tx\t{bname}\t{bchr}\t{bstart}\t{bend}\t{bpval}\t{btrig}\tna\n")
            bset.add(bname)

    # Summary
    print(f"phasis PHAS total: {len(phasisL)} | ManualCuration PHAS total: {len(ManualCurationL)}")
    print(f"phasis matched count: {amatched} | ManualCuration matched count: {bmatched}")
    print(f"phasis matched uniq: {len(aset)} | ManualCuration matched uniq: {len(bset)}")

    # Write Summary file
    sm_out.write(f"phasis file: {phasis}\n")
    sm_out.write(f"ManualCuration file: {ManualCuration}\n")
    sm_out.write(f"flanksize: {flanksize}nt\n")
    sm_out.write(f"phasis PHAS total: {len(phasisL)} | ManualCuration PHAS total: {len(ManualCurationL)}\n")
    sm_out.write(f"phasis matched count: {amatched} | ManualCuration matched count: {bmatched}\n")
    sm_out.write(f"phasis matched uniq: {len(aset)} | ManualCuration matched uniq: {len(bset)}\n")

    fh_out.close()
    sm_out.close()

    return outfile


def main():
    #revfernoL,revfernoD   = revferno_parse(revferno)
    phasisL,phasisD   = phasis_parse(phasis)

    ManualCurationL,ManualCurationD = ManualCuration_parse(ManualCuration)
    matchfile           = matchPHAS(phasisL,ManualCurationL)


if __name__ == '__main__':

    main()
    print("\nBye!!\n")
    sys.exit()


### Chage Log
## v01 -> v02
## removes "chromosome" from ManualCuration results while parsing

## v02 -> v03
## Added score/p-value to output
## Added triggers to output

