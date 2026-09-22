"""测试共享辅助：独立布局校验器。

只读取核心模块的公开数据字段，几何逻辑来自独立参考实现，
不调用被测核心的求解/变换函数。
"""

from reference import transform


def validate_layout(board, pieces, placements, expected_height):
    """独立校验一个布局。返回 None（合法）或错误描述字符串。"""
    if len(placements) != len(pieces):
        return f"放置数 {len(placements)} 与零件数 {len(pieces)} 不一致"
    occs = []
    for i, (piece, pl) in enumerate(zip(pieces, placements)):
        if pl.piece_index != i:
            return f"placements[{i}].piece_index={pl.piece_index}，应为 {i}"
        if pl.rotation not in piece.allowed_rotations:
            return f"零件 {i} 使用了未允许的旋转角 {pl.rotation}"
        if pl.mirrored and not piece.allow_mirror:
            return f"零件 {i} 使用了未允许的镜像"
        shape = transform(piece.cells, pl.rotation, pl.mirrored)
        occ = {(x + pl.x, y + pl.y) for (x, y) in shape}
        if len(occ) != len(piece.cells):
            return f"零件 {i} 占用格数异常"
        for c in occ:
            if not (0 <= c[0] < board.width and 0 <= c[1] < board.height):
                return f"零件 {i} 格子 {c} 越出板面"
            if c in board.forbidden:
                return f"零件 {i} 格子 {c} 覆盖禁放格"
        occs.append(occ)
    for i in range(len(occs)):
        for j in range(i + 1, len(occs)):
            for a in occs[i]:
                for b in occs[j]:
                    if max(abs(a[0] - b[0]), abs(a[1] - b[1])) < 2:
                        return (f"零件 {i} 与 {j} 的格子 {a}/{b} "
                                f"Chebyshev 距离 < 2（边距冲突）")
    max_y = max(c[1] for occ in occs for c in occ)
    if max_y + 1 != expected_height:
        return f"实际使用高度 {max_y + 1} 与声明 {expected_height} 不一致"
    return None
