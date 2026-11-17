from puya import log
from puya.ir import models
from puya.ir.avm_ops import AVMOp
from puya.ir.optimize.context import IROptimizationContext
from puya.ir.optimize.effects import Effect, EffectSummaries, effects_of

logger = log.get_logger(__name__)


def minimize_box_exist_asserts(
    context: IROptimizationContext, subroutine: models.Subroutine
) -> bool:
    modified = False
    for block in subroutine.body:
        if _remove_box_exists_asserts(context.effects, block):
            modified = True
    return modified


def _remove_box_exists_asserts(
    summaries: EffectSummaries | None, block: models.BasicBlock
) -> bool:
    #   for any box op that returns an exist flag that is asserted
    #   can remove any asserts after the first one as long as the box register is the same
    #   and nothing that may delete a box has run
    #   begin block
    #       exist state of all box keys are unknown
    #   box_create | box_extract | box_put | box_replace | box_resize | box_splice key
    #       set box key as known to exist
    #   anything that may delete a box: box_del, a box op on an application given by id (it can
    #   be this one), an inner transaction (another application may share box access), a call
    #   to a subroutine that may do any of these
    #       exist state of all box keys are unknown, and the exist flags read so far no longer
    #       say anything about the boxes
    #   box_get | box_len
    #       if known that box exists then its exist flag is known to be set
    #       otherwise an assert of the flag is kept, and the box exists after it
    #   assert of an exist flag
    #       if the flag is known to be set then remove the assert

    modified = False
    # keys of boxes that exist at this point
    box_exists = set[models.Value]()
    # exist flags that were read while it was unknown if the box exists, with the box key
    unknown_flags = dict[models.Value, models.Value]()
    # exist flags that are known to be set
    set_flags = set[models.Value]()
    for op in block.ops.copy():
        if isinstance(op, models.Assert):
            if op.condition in set_flags:
                logger.debug(
                    "box is known to exist, removing assert",
                    location=op.source_location,
                )
                block.ops.remove(op)
                modified = True
            elif (box_key := unknown_flags.get(op.condition)) is not None:
                # nothing may have deleted a box since the flag was read, so the box exists,
                # and it did when any later flag for it was read
                box_exists.add(box_key)
                read_since = False
                for flag, flag_box_key in unknown_flags.items():
                    read_since = read_since or flag == op.condition
                    if read_since and flag_box_key == box_key:
                        set_flags.add(flag)
            continue
        if isinstance(op, models.Assignment):
            targets = op.targets
            source: models.IRVisitable = op.source
        else:
            targets = None
            source = op
        if isinstance(source, models.BoxWrite):
            # if this deletes a box, it is to create it again
            continue
        if Effect.deletes_box in effects_of(source, summaries=summaries):
            # conservatively assume it could be deleting any box
            box_exists.clear()
            unknown_flags.clear()
        if isinstance(source, models.Intrinsic):
            match source.op:
                case (
                    AVMOp.box_create
                    | AVMOp.box_extract
                    | AVMOp.box_put
                    | AVMOp.box_replace
                    | AVMOp.box_resize
                    | AVMOp.box_splice
                ):
                    box_key = source.args[0]
                    box_exists.add(box_key)
                case AVMOp.box_get | AVMOp.box_len:
                    if targets is not None:
                        (box_key,) = source.args
                        _, exists_reg = targets
                        if box_key in box_exists:
                            set_flags.add(exists_reg)
                        else:
                            unknown_flags[exists_reg] = box_key

    return modified
