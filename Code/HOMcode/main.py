import BuildHONPlus
import BuildHON
import itertools


LastStepsHoldOutForTesting = 0
MinimumLengthForTraining = 1
InputFileDeliminator = "\t"
Verbose = True


###########################################
# Functions
###########################################


def ReadSequentialData(InputFileName):
    if Verbose:
        print("Reading raw sequential data")
    RawTrajectories = []
    with open(InputFileName) as f:
        LoopCounter = 0
        for line in f:
            fields = line.strip().split(InputFileDeliminator)
            obj = fields[0]
            trajStr = fields[1].strip("[").strip("]").split(", ")
            trajectories = [x.strip("'") for x in trajStr]

            LoopCounter += 1
            if LoopCounter % 10000 == 0:
                VPrint(LoopCounter)

            ## Other preprocessing or metadata processing can be added here

            ## Test for movement length
            MinMovementLength = MinimumLengthForTraining + LastStepsHoldOutForTesting
            if len(trajectories) < MinMovementLength:
                continue

            RawTrajectories.append([obj, trajectories])

    return RawTrajectories


def BuildTrainingAndTesting(RawTrajectories):
    VPrint("Building training and testing")
    Training = []
    Testing = []
    for trajectory in RawTrajectories:
        obj, movement = trajectory
        movement = [
            key for key, grp in itertools.groupby(movement)
        ]  # remove adjacent duplications
        if LastStepsHoldOutForTesting > 0:
            Training.append([obj, movement[:-LastStepsHoldOutForTesting]])
            Testing.append([obj, movement[-LastStepsHoldOutForTesting]])
        else:
            Training.append([obj, movement])
    return Training, Testing


def DumpRules(Rules, OutputRulesFile):
    VPrint("Dumping rules to file")
    with open(OutputRulesFile, "w") as f:
        for Source in Rules:
            for Target in Rules[Source]:
                f.write(
                    " ".join(
                        [
                            " ".join([str(x) for x in Source]),
                            "=>",
                            Target,
                            str(Rules[Source][Target]),
                        ]
                    )
                    + "\n"
                )
        f.close()
    Rules = None


def SequenceToNode(seq):
    curr = seq[-1]
    node = curr + "|"
    seq = seq[:-1]
    while len(seq) > 0:
        curr = seq[-1]
        node = node + curr + "."
        seq = seq[:-1]
    if node[-1] == ".":
        return node[:-1]
    else:
        return node


def VPrint(string):
    if Verbose:
        print(string)


def BuildHON(InputFileName, OutputNetworkFile, OutputRulesFile, MaxOrder, MinSupport):
    RawTrajectories = ReadSequentialData(InputFileName)
    TrainingTrajectory, TestingTrajectory = BuildTrainingAndTesting(RawTrajectories)
    VPrint(len(TrainingTrajectory))
    Rules = BuildHON.ExtractRules(TrainingTrajectory, MaxOrder, MinSupport)
    DumpRules(Rules, OutputRulesFile)


def BuildHONfreq(
    InputFileName, OutputNetworkFile, OutputRulesFile, MaxOrder, MinSupport
):
    Rules = None
    RawTrajectories = ReadSequentialData(InputFileName)
    TrainingTrajectory, TestingTrajectory = BuildTrainingAndTesting(RawTrajectories)
    VPrint(len(TrainingTrajectory))

    Rules = BuildHONPlus.ExtractRules(TrainingTrajectory, MaxOrder, MinSupport)
    DumpRules(Rules, OutputRulesFile)


###########################################
# Main function
###########################################

# superparameter
MaxOrder = 5
MinSupport = 1

InputFileName = "trajectories.txt"
OutputRulesFile = ""

if __name__ == "__main__":
    RawTrajectories = ReadSequentialData(InputFileName)
    TrainingTrajectory, TestingTrajectory = BuildTrainingAndTesting(RawTrajectories)
    Rules = BuildHONPlus.ExtractRules(TrainingTrajectory, MaxOrder, MinSupport)
    DumpRules(Rules, OutputRulesFile)
