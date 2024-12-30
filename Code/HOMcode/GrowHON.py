#!/usr/bin/env python
"""Module to generate a higher-order network (HON).

Exposed methods:
    grow(): grows a tree from a list of input sequences
    prune(): prunes the insignificant sequences

How to use:
    GrowHON can be used in the following ways:
    1)  Executing as a standalone script via:
        $ python growhon.py input_file output_file k
        For additional help with arguments, run:
        $ python growhon.py -h
    2) Importing as a module. See API details below.
"""

# =================================================================
# MODULES REQUIRED FOR CORE FUNCTIONALITY
# =================================================================
from collections import defaultdict, deque
from math import log, ceil
from multiprocessing import cpu_count

# =================================================================

# =================================================================
# MODULES USED FOR LOGGING AND DEBUGGING
# =================================================================
from os import getpid
from psutil import Process
from time import perf_counter, strftime, gmtime
import cpuinfo

# import logging last or other modules may adjust the logging level
import logging

# =================================================================


class HONTree:
    """Main module for growing a HON.
    Exposes three methods: grow(), prune().
    """

    def __init__(
        self,
        k,
        inf_name=None,
        num_cpus=0,
        inf_delimiter=" ",
        inf_num_prefixes=1,
        order_delimiter="|",
        prune=True,
        bottom_up=False,
        tau=1.0,
        min_support=1,
        log_name=None,
        seq_grow_log_step=1000,
        par_grow_log_interval=0.2,
        prune_log_interval=0.2,
        verbose=False,
        dotf_name=None,
    ):
        """Init method for GrowHON.

        Description:
            __init__ creates a new instance of HONTree with the given
            parameters. It first creates a root for the tree.
            If inf_name is supplied, it calls grow(). If grow() is
            called and prune==True, it then calls prune().

        Parameters:
            k (int): specifies the max order of the HON
            inf_name (str or None): input file, call grow() if provided
            num_cpus (int): specifies the number of worker processes
            inf_delimiter (char): delimiting char for input sequences
            inf_num_prefixes (int): skip this many items in each sequence
            order_delimiter (char): delimiter for higher orders
            prune (bool): whether to call prune() during init()
            tau (float): multiplier for KLD in prune()
            min_support (int): min frequency for higher order sequences
            log_name (str or None): where to write log output
            seq_grow_log_step (int): heartbeat interval for logging
            par_grow_log_interval (float): heartbeat interval as %
            prune_log_interval (float): heartbeat interval for prune()
            verbose (bool): print extra messages for debugging
        """

        logging.basicConfig(
            level=logging.INFO,
            filename=log_name,
            format="%(levelname)-8s  %(asctime)23s  "
            + "%(runtime)10s  %(rss_size)6dMB  %(message)s",
        )
        self.logger = logging.LoggerAdapter(logging.getLogger(), self._LogMap())
        try:
            self.logger.info(
                "Initializing GrowHON on CPU {} (PID {})".format(
                    cpuinfo.get_cpu_info()["brand"], getpid()
                )
            )
        except:
            self.logger.info("Initializing GrowHON; could not detect CPU info.")
        # root is essentially a dummy node
        self.root = self._HONNode(tuple())
        self.k = k
        self.nmap = defaultdict(int)
        self.inf_delimiter = inf_delimiter
        self.inf_num_prefixes = inf_num_prefixes
        self.seq_grow_log_step = seq_grow_log_step
        self.par_grow_log_interval = par_grow_log_interval
        self.prune_log_interval = prune_log_interval
        self.min_support = min_support
        self._HONNode.order_delimiter = order_delimiter
        self.tau = tau
        self.verbose = verbose

        self.num_cpus = num_cpus if num_cpus else cpu_count()
        # if ray is unavailable, default to 1 worker
        if self.num_cpus > 1:
            self.logger.warn("Ray is unavailable; parallel mode disabled.")
            self.num_cpus = 1
        if inf_name:
            self.grow(inf_name)
            if dotf_name:
                self.logger.info("Dumping divergences...")
                with open(dotf_name, "w+") as otf_divergences:
                    otf_divergences.write("\n".join(self.get_divergences()))
            if prune:
                self.prune(bottom_up)

    # =================================================================
    # BEGIN EXPOSED FUNCTIONS FOR GROWING, PRUNING
    # =================================================================
    def close_log(self):
        logging.shutdown()

    def grow(self, inf_name, num_cpus=None, inf_delimiter=None, inf_num_prefixes=None):
        """The main method used to grow the HONTree.

        Description:
            This method delegates to either _grow_sequential or
            _grow_parallel, depending on the value of num_cpus. It
            first checks for a user-supplied value. If there is none,
            it calls multiprocessing.cpu_count().

        Parameters:
            inf_name (str): input file, required
            num_cpus (int): number of worker processes to initiate
            inf_delimiter (char): delimiter for input sequences
            inf_num_prefixes (char): items to skip in input sequences

        Returns:
            None
        """
        # reassign parameters if they were passed explicitly
        if num_cpus:
            self.num_cpus = num_cpus
        if inf_delimiter:
            self.inf_delimiter = inf_delimiter
        if inf_num_prefixes:
            self.inf_num_prefixes = inf_num_prefixes

        self.logger.info(
            "Growing tree with {} and max order {}...".format(
                str(self.num_cpus) + " worker" + ("s" if self.num_cpus > 1 else ""),
                self.k,
            )
        )
        if self.num_cpus > 1:
            self._grow_parallel(inf_name)
        else:
            self._grow_sequential(inf_name)
        self.logger.info("Growing successfully completed!")
        if self.verbose:
            self.logger.info("Tree dump:\n{}".format(self))

    def prune(self, bottom_up=False, tau=None, min_support=None):
        """Uses statistical methods to prune the grown HONTree.

        Description:
            This method checks all nodes in the tree (except those in
            the first and last levels) and uses relative entropy
            to determine whether the higher order or lower order
            sequences should be preserved. Nodes that are not preserved
            have their in-degrees reduced, possibly to 0.

        Parameters:
            bottom_up (bool): perform pruning on bottom levels first
            tau (float): multiplier for KLD threshold
            min_support (int): min frequency for higher order sequences

        Returns:
            None
        """
        # reassign parameters if they were passed explicitly
        if tau:
            self.tau = tau
        if min_support:
            self.min_support = min_support

        self._prune_top_down()
        if self.verbose:
            self.logger.info("Tree dump:\n{}".format(self))

    # =================================================================
    # END MAIN FUNCTIONS
    # =================================================================

    # =================================================================
    # BEGIN SEQUENTIAL MODE FUNCTIONS
    # =================================================================
    # if the child does not already exist, add it
    # returns a reference to the child node
    def _add_child(self, parent, child_label):
        """Insert a sequence into the tree.

        Description:
            If the sequence does not yet exist in the tree, we first
            create a new _HONNode object. If it does exist, we can
            simply increment its in-degree and its parent's out-degree.
            All nodes are also referenced in nmap by their unique label
            to enable fast lookup to lower-orders.

        Parameters:
            parent (_HONNode): reference to the parent of the new child

        Returns:
            child (_HONNode): reference to the child node
        """
        if child_label in parent.children:
            child = parent.children[child_label]
        else:
            child = self._HONNode(parent.name + (child_label,), parent)
            parent.children[child_label] = child
            if child.order < self.k:
                self.nmap[child.name] = child
        parent.out_deg += 1
        child.in_deg += 1
        return child

    def _grow_sequential(self, inf_name):
        """Process an entire input file using 1 worker.

        Description:
            This method iterates over each line in an input file,
            and calls the _grow_sequence method for each one.

        Parameters:
            inf_name (str): the input file

        Returns:
            None (the tree is modified in-place and accessed from root)
        """

        with open(inf_name, "r") as inf:
            for i, line in enumerate(inf):
                if not i % self.seq_grow_log_step:
                    self.logger.info("Processing input line {:,d}...".format(i))
                self._grow_sequence(line)

    def _grow_sequence(self, seq):
        """Process a single sequence from the input.

        Description:
            This method processes a single input sequence by generating
            all sequential combinations up to length k. It
            does include the "tail" of the sequence, meaning the
            sequences at the very end of the sequence which may have
            a shorter length. It passes each sequence to the _add_child
            method to insert into the tree.

        Parameters:
            line (str): an input sequence

        Returns:
            None
        """
        # we build the tree 1 level higher than k
        # so k + 1 is required
        q = deque(maxlen=self.k + 1)
        seq = seq.strip().split(self.inf_delimiter)[self.inf_num_prefixes :]
        # prime the queue
        # e.g. we have line=='1 2 3 4 5' && k==2 -> q=[1,2,3]
        for e in seq[: self.k + 1]:
            q.append(e)
        # loop through the rest of the sequence
        for e in seq[self.k + 1 :]:
            # for each combination, add the sequence to the tree
            # e.g. q==[1,2,3]
            #   -> insert 1 as child of root
            #   -> insert 2 as child of root->1
            #   -> insert 3 as child of root->1->2
            cur_parent = self.root
            for node in q:
                cur_parent = self._add_child(cur_parent, node)
            # move forward in the sequence
            # e.g. q==[1,2,3] & next item in line==4 -> q==[2,3,4]
            q.popleft()
            q.append(e)
        # add the "tail" of the sequence
        # this could be ommitted if you want to truncate the sequences
        while q:
            cur_parent = self.root
            for node in q:
                cur_parent = self._add_child(cur_parent, node)
            q.popleft()

    # =================================================================
    # END SEQUENTIAL MODE FUNCTIONS
    # =================================================================

    # =================================================================
    # BEGIN SUPPORTING (_ and __) FUNCTIONS
    # =================================================================
    def __str__(self):
        """Calls a recursive helper to traverse and print the tree."""
        return self.__str_helper(self.root)

    def __str_helper(self, node):
        """A recursive DFS helper for printing the tree."""
        if node:
            s = node.dump() if node != self.root else ""
            for child in node.children.values():
                s += self.__str_helper(child)
            return s

    def _delete_children(self, node):
        """Reduces the incoming node's in-degree to 0.
        Also reduces its parent's out-degree by the same number.
        The node is not deleted from memory.

        Parameters:
            node (_HONNode): the node to delete
        """
        node.out_deg = 0
        for c in node.children.values():
            c.in_deg = 0

    def _delete_node(self, node):
        """Reduces the incoming node's in-degree to 0.
        Also reduces its parent's out-degree by the same number.
        The node is not deleted from memory.

        Parameters:
            node (_HONNode): the node to delete
        """
        node.parent.out_deg -= node.in_deg
        node.in_deg = 0

    def _has_dependency(self, hord_node):
        """Decides whether a higher-order node should be preserved.

        Description:
            This method measures the KLD (Kullback-Leibler Divergence
            or relative entropy) of a higher-order node with respect
            to its lower-order counterpart. If the KLD exceeds
            the threshold as defined by _get_divergence_threshold,
            the higher-order node is preserved and the lower-order
            pruned. If not, the higher-order node is pruned. The higher
            order node must also exceed the value of min_support.

        Parameters:
            hord_node (_HONNode): the higher-order node to check

        Returns:
            True if the higher-order node should be preserved;
            False otherwise
        """
        if hord_node.in_deg >= self.min_support:
            divergence = self._get_divergence(hord_node)
            threshold = self._get_divergence_threshold(hord_node)
            return divergence > threshold
        """
        if self.verbose: 
            self.logger.info('{} -> {}'
                        .format(str(hord_node), str([(k, v.in_deg) 
                        for k, v in hord_node.children.items()])))
            self.logger.info('{} -> {}'
                        .format(str(self._get_lord_match(hord_node)), str([(k, v.in_deg)
                        for k, v in self._get_lord_match(hord_node).children.items()])))
            self.logger.info('dvrg:{} ; thrs: {}'.format(divergence, threshold))
            if divergence > threshold:
                self.logger.info('{} has a dependency.'.format(str(hord_node)))
            else:
                self.logger.info('{} does not have a dependency.'.format(str(hord_node)))
        """
        return False

    def _get_divergence(self, hord_node):
        """Measures the KLD (relative entropy) of a higher-order node.

        Parameters:
            hord_node (_HONNode): the higher-order node to check

        Returns:
            The KLD (Kullback-Leibler divergence or relative entropy)
            of the higher-order node, as a floating point number.
        """
        lord_node = self._get_lord_match(hord_node)
        divergence = 0.0
        for child_label, child in hord_node.children.items():
            hord_conf = child.in_deg / hord_node.out_deg
            divergence += hord_conf * log(
                (hord_conf * lord_node.out_deg)
                / lord_node.children[child_label].in_deg,
                2,
            )
        return divergence

    def _get_divergence_threshold(self, hord_node):
        """Determines the KLD threshold for a higher-order node.

        Parameters:
            hord_node (_HONNode): the higher-order node to check

        Returns:
            The threshold the node's KLD must exceed to be preserved.
        """
        return self.tau * hord_node.order / log(1 + hord_node.in_deg, 2)

    def _get_lord_match(self, hord_node):
        """Used to find a node's lower-order counterpart.

        Description:
            This method is used by several others to find a
            higher-order node's lower-order counterpart in O(1) time.
            This is done by truncating the first element (oldest
            history) from the higher-order node's label, then finding
            the new label in the nmap.

        Parameters:
            hord_node (_HONNode): the higher-order node

        Returns:
            (_HONNode) the object reference to the lower-order node
        """
        return self.nmap[hord_node.name[1:]] if hord_node.order else None

    def _prune_top_down(self):
        """Prune the tree from the top down, starting with higher orders.

        Description:
            Top-down pruning is the default because it is more comprehensive
            than bottom-up. Its accuracy with respect to top-down pruning is
            still unverified, so it may be worthwhile to try both.

        Parameters:
            None

        Returns:
            None
        """
        self.logger.info("Pruning tree top-down...")
        # starting at the second-to-last level of the tree
        for order in range(self.k - 1, 0, -1):
            # getting nodes from nmap is faster than tree traversal
            cur_order = [n for n in self.nmap.values() if n.order == order]
            log_step = ceil(self.prune_log_interval * len(cur_order))
            self.logger.info(
                "Pruning {:,d} nodes from order {}...".format(len(cur_order), order + 1)
            )
            for i, node in enumerate(cur_order):
                # add an update to the log every so often
                if not i % log_step:
                    self.logger.info(
                        "Pruning node {:,d}/{:,d} ({}%) in order {}...".format(
                            i, len(cur_order), int(i * 100 / len(cur_order)), order + 1
                        )
                    )
                # the two criteria for preserving a higher-order node are if
                # 1) it has a higher-order descendent (to preserve flow)
                # or
                # 2) it has a dependency, as indicate by its relative entropy
                if node.marked or (self._has_dependency(node)):
                    node.parent.marked = True
                else:
                    self._delete_children(node)
            self.logger.info(
                "Pruning successfully completed on order {}.".format(order + 1)
            )
        self.logger.info("Pruning successfully completed all orders!")

    # =================================================================
    # END SUPPORTING FUNCTIONS
    # =================================================================

    # =================================================================
    # BEGIN _HONNode DEFINITION
    # =================================================================
    class _HONNode:
        """The definition of all objects inserted into HONTree.

        Description:
            Each _HONNode represents a sequence from the input data and
            an edge in the output HON.

        Static Variables:
            order_delimiter (char): for labelling higher order nodes
        """

        # initialize to -1 to discount root node
        order_delimiter = None

        def __init__(self, name="", parent=None):
            self.name = name
            self.order = len(self.name) - 1
            self.in_deg = 0
            self.out_deg = 0
            self.parent = parent
            self.children = {}
            self.marked = False  # used during pruning
            self.checked_for_merge = False

        def __str__(self):
            return "{}[{}:{}]".format(self.get_label_full(), self.in_deg, self.out_deg)

        def dump(self):
            return "------->" * (len(self.name) - 1) + str(self) + "\n"

        def get_label_full(self):
            return HONTree._HONNode.order_delimiter.join(
                reversed([str(c) for c in self.name])
            )

    # =========================================================================
    # END _HONNode DEFINITION
    # =========================================================================

    # =========================================================================
    # BEGIN _LogMap DEFINITION
    # =========================================================================
    class _LogMap:
        """Internal class used to standardize logging."""

        def __init__(self):
            self.start_time = perf_counter()
            self._info = {}
            self._info["runtime"] = self.get_time_seconds
            self._info["rss_size"] = self.get_rss

        def __getitem__(self, key):
            return self._info[key]()

        def __iter__(self):
            return iter(self._info)

        def get_time(self):
            return strftime("%H:%M:%S", gmtime(perf_counter() - self.start_time))

        def get_time_seconds(self):
            return "{:.3f}".format(perf_counter() - self.start_time)

        def get_rss(self):
            return Process(getpid()).memory_info().rss >> 20

    # =========================================================================
    # END _LogMap DEFINITION
    # =========================================================================


# =================================================================
# END HONTree DEFINITION
# =================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("infname", help="source path + file name")
    parser.add_argument(
        "otfname", help="destination path + file name, use ! to avoid save"
    )
    parser.add_argument("k", help="max order to use in growing the HON", type=int)
    parser.add_argument(
        "-w", "--numcpus", help="number of workers", type=int, default=1
    )
    parser.add_argument(
        "-p",
        "--infnumprefixes",
        help="number of prefixes for input sequences",
        type=int,
        default=1,
    )
    parser.add_argument(
        "-di",
        "--infdelimiter",
        help="delimiter for entities in input sequences",
        default=" ",
    )
    parser.add_argument(
        "-do", "--otfdelimiter", help="delimiter for output network", default=" "
    )
    parser.add_argument(
        "-s",
        "--skipprune",
        help="whether to skip pruning the tree",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-b",
        "--bottomup",
        help="enable bottom-up pruning (default is top-down)",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-t",
        "--tau",
        help="threshold multiplier for determining dependencies",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "-m",
        "--minsupport",
        help="minimum support required for a dependency",
        type=int,
        default=1,
    )
    parser.add_argument(
        "-e", "--dotfname", help="destination path + file for divergences", default=None
    )
    parser.add_argument(
        "-o", "--logname", help="location to write log output", default=None
    )
    parser.add_argument(
        "-lsg",
        "--logisgrow",
        help="logging interval for sequential growth",
        type=int,
        default=10000,
    )
    parser.add_argument(
        "-lpg",
        "--logipgrow",
        help="logging interval for parallel growth",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "-lpr",
        "--logiprune",
        help="logging interval for pruning",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="print more messages for debugging",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()

    t1 = HONTree(
        args.k,
        args.infname,
        inf_num_prefixes=args.infnumprefixes,
        inf_delimiter=args.infdelimiter,
        num_cpus=args.numcpus,
        log_name=args.logname,
        verbose=args.verbose,
        prune=not args.skipprune,
        bottom_up=args.bottomup,
        tau=args.tau,
        min_support=args.minsupport,
        seq_grow_log_step=args.logisgrow,
        par_grow_log_interval=args.logipgrow,
        prune_log_interval=args.logiprune,
        dotf_name=args.dotfname,
    )
    t1.close_log()
# =====================================================================
# END growhon.py
# =====================================================================
