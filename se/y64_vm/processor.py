#!/usr/bin/env python3
# A virtual machine for the Y64 architecture (CSAPP chapter 4)
# refernece:
# page393 of CSAPP 3rd edition

import operator
import random
import re
from dataclasses import dataclass
from typing import Dict, Tuple, Union

__all__ = [
    "Processor",
    "num_to_hex",
    "hex_to_num",
    "oprand_to_bytecode",
    "bytecode_to_instruction",
    "program_to_bytecode",
    "name_to_instruction",
    "code_to_instruction",
]


def num_to_hex(num: int, byte_length: int = 8, order: str = "big") -> str:
    assert order in ("big", "little"), "order should be 'big' or 'little'"
    hex_bytes = num.to_bytes(byte_length, byteorder=order)
    return hex_bytes.hex()


def hex_to_num(hex_str: str, order: str = "big") -> int:
    assert order in ("big", "little"), "order should be 'big' or 'little'"
    hex_bytes = bytes.fromhex(hex_str)
    return int.from_bytes(hex_bytes, byteorder=order)


_REG_MAP = {
    "rsp": 0, "rbp": 1, "rsi": 2, "rdi": 3,
    "rax": 4, "rbx": 5, "rcx": 6, "rdx": 7,
}


def oprand_to_bytecode(oprand: str) -> Union[Tuple[str, str], str]:
    """Convert oprand to bytecode

    Supported format: $123, %rax, 123(%rax)
    """
    oprand = oprand.strip()
    data = ""
    if oprand.startswith("$"):  # immediate value
        value = int(oprand[1:])
        return num_to_hex(value)
    elif oprand.startswith("%"):  # register
        reg_name = oprand[1:]
        if reg_name in _REG_MAP:
            return str(_REG_MAP[reg_name])
    elif "(" in oprand and ")" in oprand:  # memory address
        data, reg = oprand.split("(")
        data = data.strip()
        data_value = int(data) if data else 0
        reg_name = reg.split(")")[0].strip().split("%")[-1].strip()  # maybe regex is better
        return str(_REG_MAP[reg_name]), num_to_hex(data_value)
    else:  # label like jle loop
        return f"[offset+{oprand}]"


def bytecode_to_register(bytecode: str) -> str:
    code = hex_to_num("0" + bytecode)
    reg_name = ""
    for key, value in _REG_MAP.items():
        if value == code:
            reg_name = key
            break

    return f"%{reg_name}" if reg_name else f"%r{code}"


@dataclass
class Instruction:

    code: str
    code_length: int = 1
    with_data: bool = False

    def fetch(self, processor: "Processor"):
        code = processor.read_code(self.code_length)
        assert code.startswith(self.code)
        self.byte_code = code

        if self.with_data:
            data_length = processor.byte_length
            data = processor.read_code(byte_length=data_length + self.code_length)
            self.data = data[-data_length * 2:]
        else:
            self.data = None
            data_length = 0

        pc = processor.pc + (self.code_length + data_length)
        self.valP = pc  # address of next instruction, store to update

    def decode(self, processor: "Processor"):
        pass

    def execution(self, processor: "Processor"):
        pass

    def memory(self, processor: "Processor"):
        pass

    def write_back(self, processor: "Processor"):
        pass

    def pc_update(self, processor: "Processor"):
        processor.pc = self.valP

    def reset(self, processor: "Processor"):
        # TODO: check if this function is necessary
        self.valP = None
        self.data = None
        processor.reset_flags()

    def to_byte_code(self, arg_text: str) -> str:
        arg_text = arg_text.strip()
        if not arg_text:  # no operand needed, like halt/nop/ret
            return self.code

        if "," in arg_text:
            rA, rB = arg_text.split(",")
            rA = oprand_to_bytecode(rA)
            rB = oprand_to_bytecode(rB)
        else:  # single operand
            rA = oprand_to_bytecode(arg_text)
            rB = "F"  # like pushq/popq

        code = self.code
        if self.code_length > 1:
            code += f"{rA}{rB}"  # works only with_data = False
        return code

    def byte_to_assembly(self, byte_code: str):
        obj_name = self.__class__.__name__.lower()
        assert byte_code.startswith(self.code), f"invalid byte_code: {byte_code}"
        assembly = obj_name

        if self.code_length > 1:
            rA, rB = bytecode_to_register(byte_code[2]), bytecode_to_register(byte_code[3])
            assembly += f" {rA}, {rB}"
        if self.with_data:
            data = "$" + str(hex_to_num(byte_code[-16:]))
            assembly += f" {data}"
        return assembly

    @property
    def byte_length(self):
        data_length = 8 if self.with_data else 0
        return self.code_length + data_length


@dataclass
class Halt(Instruction):
    # stop the processor
    code: str = "00"


@dataclass
class Nop(Instruction):
    # do nothing
    code: str = "10"


@dataclass
class RRmovq(Instruction):
    # register to register move
    # rrmovq rA, rB
    code: str = "20"
    code_length: int = 2

    def fetch(self, processor):
        super().fetch(processor)
        self.rA = self.byte_code[2]
        self.rB = self.byte_code[3]

    def decode(self, processor):
        self.valA = processor.read_register(self.rA)

    def execution(self, processor):
        self.valE = self.valA + 0

    def write_back(self, processor):
        processor.set_register(self.rB, self.valE)


@dataclass
class IRmovq(Instruction):
    # immediate to register move
    # irmovq V, rB
    code: str = "30"
    code_length: int = 2
    with_data: bool = True

    def fetch(self, processor):
        super().fetch(processor)
        assert self.byte_code[2] == "F"
        self.rB = self.byte_code[3]
        self.data = hex_to_num(self.data)

    def execution(self, processor):
        self.valE = self.data + 0

    def write_back(self, processor):
        processor.set_register(self.rB, self.valE)

    def to_byte_code(self, arg_text: str) -> str:
        arg_text = arg_text.strip()
        code = self.code

        assert "," in arg_text, f"invalid arg_text: {arg_text}"
        data, rB = arg_text.split(",")
        data = oprand_to_bytecode(data)
        rB = oprand_to_bytecode(rB)
        code += "F" + rB + data
        return code

    def byte_to_assembly(self, byte_code):
        ass_code = super().byte_to_assembly(byte_code)
        inst, _, rB, data = ass_code.split(" ")
        return f"{inst} {data}, {rB}"


@dataclass
class RMmovq(Instruction):
    # register to memory move
    # rmovq rA, D(rB)
    code: str = "40"
    with_data: bool = True
    code_length: int = 2

    def fetch(self, processor):
        super().fetch(processor)
        self.rA = self.byte_code[2]
        self.rB = self.byte_code[3]
        self.data = hex_to_num(self.data)

    def decode(self, processor):
        self.valA = processor.read_register(self.rA)
        self.valB = processor.read_register(self.rB)

    def execution(self, processor):
        self.valE = self.valB + self.data

    def memory(self, processor):
        processor.memory[self.valE] = self.valA

    def to_byte_code(self, arg_text: str) -> str:
        arg_text = arg_text.strip()
        code = self.code

        assert "," in arg_text, f"invalid arg_text: {arg_text}"
        rA, rB = arg_text.split(",")
        rA = oprand_to_bytecode(rA)
        rB, data = oprand_to_bytecode(rB)
        code += rA + rB + data
        return code

    def byte_to_assembly(self, byte_code):
        ass_code = super().byte_to_assembly(byte_code)
        inst, rA, rB, data = ass_code.split()
        if data.startswith("$"):
            data = data[1:]
        data = int(data)
        data = f"({rB})" if data == 0 else f"{data}({rB})"
        return f"{inst} {rA} {data}"


@dataclass
class MRmovq(Instruction):
    # memory to register move
    # mrmovq D(rB), rA
    code: str = "50"
    code_length: int = 2
    with_data: bool = True

    def fetch(self, processor: "Processor"):
        super().fetch(processor)
        self.rA = self.byte_code[2]
        self.rB = self.byte_code[3]
        self.data = hex_to_num(self.data)

    def decode(self, processor: "Processor"):
        self.valA = processor.read_register(self.rA)
        self.valB = processor.read_register(self.rB)

    def execution(self, processor: "Processor"):
        self.valE = self.valB + self.data

    def memory(self, processor: "Processor"):
        processor.set_register(self.valA, processor.memory[self.valE])

    def to_byte_code(self, arg_text: str) -> str:
        arg_text = arg_text.strip()
        code = self.code

        assert "," in arg_text, f"invalid arg_text: {arg_text}"
        rB, rA = arg_text.split(",")
        rA = oprand_to_bytecode(rA)
        rB, data = oprand_to_bytecode(rB)
        code += rA + rB + data
        return code

    def byte_to_assembly(self, byte_code):
        ass_code = super().byte_to_assembly(byte_code)
        inst, rA, rB, data = ass_code.split()
        if rA.endswith(","):
            rA = rA[:-1]
        if data.startswith("$"):
            data = data[1:]
        data = int(data)
        data = f"({rB})" if data == 0 else f"{data}({rB})"
        return f"{inst} {data}, {rA}"


@dataclass
class Opq(Instruction):
    # operation of two registers
    # opq rA, rB
    code: str = "6"
    # addq(60), subq(61), andq(62), xorq(63)
    code_length: int = 2

    def code_to_func(self, code):
        return {
            "0": operator.add,
            "1": operator.sub,
            "2": operator.and_,
            "3": operator.xor,
        }[code]

    def fetch(self, processor: "Processor"):
        super().fetch(processor)
        self.func = self.code_to_func(self.byte_code[1])
        self.rA = self.byte_code[2]
        self.rB = self.byte_code[3]

    def decode(self, processor: "Processor"):
        self.valA = processor.read_register(self.rA)
        self.valB = processor.read_register(self.rB)

    def execution(self, processor: "Processor"):
        self.valE = self.func(self.valA, self.valB)
        processor.set_flags(self.valE)

    def write_back(self, processor: "Processor"):
        processor.set_register(self.rB, self.valE)

    def byte_to_assembly(self, byte_code):
        ass_code = super().byte_to_assembly(byte_code)
        _, args = ass_code.split(maxsplit=1)
        inst_name = {
            "0": "addq", "1": "subq",
            "2": "andq", "3": "xorq",
        }[byte_code[1]]
        return f"{inst_name} {args}"


@dataclass
class Jxx(Instruction):
    # jmp(70), jle(71), jl(72), je(73), jne(74), jge(75), jg(76)
    code: str = "7"
    with_data: bool = True

    def fetch(self, processor: "Processor"):
        super().fetch(processor)
        self.func = self.byte_code[1]
        self.data = hex_to_num(self.data)

    def execution(self, processor):
        flags = processor.flags
        zf, sf, of = flags["ZF"], flags["SF"], flags["OF"]

        if self.func == "0":
            # direct jump
            self.cnd = True
        elif self.func == "1":
            # less or equal
            self.cnd = zf or (sf != of)
        elif self.func == "2":
            # less
            # sf = 1, of = 0, a - b < 0
            # sf = 1, of = 0, a - b < 0
            self.cnd = (sf != of)
        elif self.func == "3":
            # equal
            self.cnd = zf
        elif self.func == "4":
            # not equal
            self.cnd = not zf
        elif self.func == "5":
            # greater or equal
            self.cnd = zf or (sf == of)
        elif self.func == "6":
            # greater
            # sf = 1, of = 1, a - b > 0
            # sf = 0, of = 0, a - b > 0
            self.cnd = (sf == of)
        else:
            raise ValueError(f"unknown function code: {self.func}")

    def pc_update(self, processor):
        next_pc = self.data if self.cnd else self.valP
        processor.pc = next_pc

    def to_byte_code(self, arg_text: str) -> str:
        arg_text = arg_text.strip()
        code = self.code

        assert "," not in arg_text, f"invalid arg_text: {arg_text}"
        data = oprand_to_bytecode(arg_text)
        code += data
        return code

    def byte_to_assembly(self, byte_code):
        ass_code = super().byte_to_assembly(byte_code)


@dataclass
class Cmovxx(Instruction):
    # conditional move (register to register)
    # rrmovq(20), cmovle(21), cmovl(22), cmove(23), cmovne(24), cmovge(25), cmovg(26)
    code: str = "2"
    code_length: int = 2

    def fetch(self, processor: "Processor"):
        super().fetch(processor)
        self.func = self.byte_code[1]
        self.rA = self.byte_code[2]
        self.rB = self.byte_code[3]

    def decode(self, processor: "Processor"):
        self.valA = processor.read_register(self.rA)

    def execution(self, processor: "Processor"):
        # The same as Jxx
        flags = processor.flags
        zf, sf, of = flags["ZF"], flags["SF"], flags["OF"]

        if self.func == "1":  # less or equal
            self.cnd = zf or (sf != of)
        elif self.func == "2":  # less
            self.cnd = (sf != of)
        elif self.func == "3":  # equal
            self.cnd = zf
        elif self.func == "4":  # not equal
            self.cnd = not zf
        elif self.func == "5":  # greater or equal
            self.cnd = zf or (sf == of)
        elif self.func == "6":  # greater
            self.cnd = (sf == of)
        else:
            raise ValueError(f"unknown function code: {self.func}")

    def write_back(self, processor: "Processor"):
        if self.cnd:
            processor.set_register(self.rB, self.valA)


@dataclass
class Call(Instruction):
    # call + direct address
    # call Dest
    code: str = "80"
    with_data: bool = True

    def decode(self, processor: "Processor"):
        self.valB = processor.read_register("rsp")

    def execution(self, processor: "Processor"):
        self.valE = self.valB - processor.byte_length

    def memory(self, processor: "Processor"):
        processor.memory[self.valE] = self.valP

    def write_back(self, processor: "Processor"):
        processor.set_register("rsp", self.valE)

    def pc_update(self, processor: "Processor"):
        processor.pc = hex_to_num(self.data)

    def to_byte_code(self, arg_text: str) -> str:
        arg_text = arg_text.strip()
        code = self.code

        assert "," not in arg_text, f"invalid arg_text: {arg_text}"
        data = oprand_to_bytecode(arg_text)
        code += data
        return code


@dataclass
class Ret(Instruction):
    code: str = "90"

    def decode(self, processor: "Processor"):
        self.valB = processor.read_register("rsp")

    def execution(self, processor: "Processor"):
        self.valE = self.valB + processor.byte_length

    def memory(self, processor: "Processor"):
        self.valM = processor.memory[self.valB]

    def write_back(self, processor: "Processor"):
        processor.set_register("rsp", self.valE)

    def pc_update(self, processor: "Processor"):
        processor.pc = hex_to_num(self.valM)


@dataclass
class Pushq(Instruction):
    # pushq rA
    code: str = "A0"
    code_length: int = 2

    def fetch(self, processor: "Processor"):
        super().fetch(processor)
        self.rA = self.byte_code[2]
        assert self.byte_code[3] == "F"

    def decode(self, processor: "Processor"):
        self.valA = processor.read_register(self.rA)
        self.valB = processor.read_register("rsp")

    def execution(self, processor: "Processor"):
        # stack grows downward
        self.valE = self.valB - processor.byte_length

    def memory(self, processor: "Processor"):
        processor.memory[self.valE] = self.valA

    def write_back(self, processor: "Processor"):
        processor.set_register("rsp", self.valE)

    def byte_to_assembly(self, byte_code):
        ass_code = super().byte_to_assembly(byte_code)
        inst, rA, _ = ass_code.split()
        if rA.endswith(","):
            rA = rA[:-1]
        return f"{inst} {rA}"


@dataclass
class Popq(Instruction):
    # popq rA
    code: str = "B0"
    code_length: int = 2

    def fetch(self, processor: "Processor"):
        super().fetch(processor)
        self.rA = self.byte_code[2]
        assert self.byte_code[3] == "F"

    def decode(self, processor: "Processor"):
        self.valA = processor.read_register("rsp")
        self.valB = processor.read_register("rsp")

    def execution(self, processor: "Processor"):
        self.valE = self.valB + processor.byte_length

    def memory(self, processor: "Processor"):
        self.valM = processor.memory[self.valA]

    def write_back(self, processor: "Processor"):
        processor.set_register("rsp", self.valE)
        # NOTE: valM is prior to valE, consider popq %rsp
        # the rsp value should be stack value in the end, but not the next stack address
        processor.set_register(self.rA, self.valM)

    def byte_to_assembly(self, byte_code):
        ass_code = super().byte_to_assembly(byte_code)
        inst, rA, _ = ass_code.split()
        if rA.endswith(","):
            rA = rA[:-1]
        return f"{inst} {rA}"


halt = Halt()
nop = Nop()
rrmovq = RRmovq()
irmovq = IRmovq()
rmmovq = RMmovq()
mrmovq = MRmovq()
addq = Opq(code="60")
subq = Opq(code="61")
andq = Opq(code="62")
xorq = Opq(code="63")
jmp = Jxx(code="70")
jle = Jxx(code="71")
jl = Jxx(code="72")
je = Jxx(code="73")
jne = Jxx(code="74")
jge = Jxx(code="75")
jg = Jxx(code="76")
cmovxx = Cmovxx()
call = Call()
ret = Ret()
pushq = Pushq()
popq = Popq()

INST_MAP = {
    x.code: x
    for x in [
        halt, nop,
        addq, subq, andq, xorq,
        rrmovq, irmovq, rmmovq, mrmovq,
        jmp, jle, jl, je, jne, jge, jg,
        cmovxx, call, ret, pushq, popq,
    ]
}

INST_TEXT_MAP = {
    "halt": halt, "nop": nop,
    "rrmovq": rrmovq, "irmovq": irmovq, "rmmovq": rmmovq, "mrmovq": mrmovq,
    "addq": addq, "subq": subq, "andq": andq, "xorq": xorq,
    "jmp": jmp, "jle": jle, "jl": jl, "je": je, "jne": jne, "jge": jge, "jg": jg,
    "call": call, "ret": ret, "pushq": pushq, "popq": popq,
}


def code_to_instruction(code: str):
    value = INST_MAP.get(code, INST_MAP.get(code[0], halt))
    if isinstance(value, Halt) and code and code != "00":
        print(f"Warning: unknown code {code}, fallback to halt")
    return value


def name_to_instruction(name: str):
    return INST_TEXT_MAP.get(name, halt)


class Memory:

    def __init__(self, size: int, byte_length: int = 8):
        self.size = size
        self.byte_length = byte_length
        assert self.size % self.byte_length == 0, f"size{size} % bytes({byte_length}) != 0"
        self.mem = "".join(["00" for _ in range(size)])

    def read_byte(self, addr: int, byte_length: int = 1) -> str:
        start = addr * 2
        end = start + byte_length * 2
        return self.mem[start:end]

    def partition(self) -> int:
        # TODO: add heap and data segmention
        self.stack_size = self.size // 4
        return self.stack_size

    def double_slice(self, s: slice) -> slice:
        start, stop, step = s.start, s.stop, s.step
        if start is not None:
            start = start * 2
        if stop is not None:
            stop = stop * 2
        if step is not None:
            step = step * 2
        return slice(start, stop, step)

    def __getitem__(self, key: Union[int, str, slice]) -> str:
        if isinstance(key, slice):
            slice2 = self.double_slice(key)
            return self.mem[slice2]

        addr = int(key) * 2
        return self.mem[addr:addr + self.byte_length * 2]

    def __setitem__(self, key: Union[int, str, slice], value: Union[int, str]):
        if isinstance(value, str):
            set_value = value
        else:
            set_value = num_to_hex(value, byte_length=self.byte_length)

        if isinstance(key, slice):
            addr = key.start * 2
            if key.stop is None:
                addr_end = addr + len(set_value)
            else:
                addr_end = key.stop * 2
                assert addr_end - addr == len(set_value), f"length of {set_value} != {key}"
        else:
            addr = int(key) * 2
            addr_end = addr + len(set_value)

        self.mem = self.mem[:addr] + set_value + self.mem[addr_end:]


class Processor:

    def __init__(self, num_registers: int, memory_size: int, bit: int = 64):
        assert bit % 8 == 0, "bit should be multiple of 8"
        self.byte_length = bit // 8
        self.memory = Memory(memory_size, self.byte_length)

        self.pc: int = 0  # program counter
        self.registers = {
            "rsp": 0,  # stack pointer
            "rbp": 0,  # base pointer
            "rsi": 0,  # source index
            "rdi": 0,  # destination index
            # general purpose registers
            "rax": 0,
            "rbx": 0,
            "rcx": 0,
            "rdx": 0,
        }
        self.register_map = {
            0: "rsp", 1: "rbp", 2: "rsi", 3: "rdi",
            4: "rax", 5: "rbx", 6: "rcx", 7: "rdx",
        }
        reg_length = len(self.registers)
        for i in range(num_registers):
            idx = i + reg_length
            self.registers[f"r{idx}"] = 0
            self.register_map[idx] = f"r{idx}"

        self.flags = {
            "ZF": False,  # zero flag
            "SF": False,  # sign flag
            "OF": False,  # overflow flag
        }
        self.byte_code = ""
        self.stack_size = self.memory.partition()

        self.bit = bit

        self.unsign_max = (1 << bit) - 1
        self.unsign_min = 0
        self.mask = 1 << (bit - 1)
        self.sign_max = self.mask - 1
        self.sign_min = -self.mask

    def set_flags(self, val: int):
        if val == 0:
            self.flags["ZF"] = True
        else:
            self.flags["ZF"] = False

        if val & self.mask:
            self.flags["SF"] = True
        else:
            self.flags["SF"] = False

        if val > self.sign_max or val < self.sign_min:
            self.flags["OF"] = True
        else:
            self.flags["OF"] = False

    def reset_flags(self):
        for key in self.flags:
            self.flags[key] = False

    def reset(self):
        self.reset_flags()
        self.pc = 0

    def read_code(self, byte_length: int = 1):
        # read a byte from the bin, but not increment the pc
        return self.memory[self.pc: self.pc + byte_length]

    def read_register(self, idx: Union[int, str]) -> int:
        if isinstance(idx, str) and not idx.isdigit():
            value = self.registers[idx]  # register name, directly
        else:
            idx = int(idx)
            value = self.registers[self.register_map[idx]]
        return value if isinstance(value, int) else hex_to_num(value)

    def set_register(self, idx: Union[int, str], value: int):
        if isinstance(idx, str) and not idx.isdigit():
            name = idx
        else:
            idx = int(idx)
            name = self.register_map[idx]
        self.registers[name] = value  # register name, directly

    def load_byte_code(self, byte_code: str, start_addr: int = 0):
        num_bytes = len(byte_code) // 2
        # A new program is loaded into memory, and the pc/rsp should be reset
        self.registers["rsp"] = self.memory.size - 1
        self.pc = start_addr
        self.memory[start_addr: start_addr + num_bytes] = byte_code

    def run_one_step(self):
        code = self.read_code(byte_length=1)
        instruction = code_to_instruction(code)

        instruction.fetch(self)
        instruction.decode(self)
        instruction.execution(self)
        instruction.memory(self)
        instruction.write_back(self)
        instruction.pc_update(self)

        return instruction

    def run(self):
        code = self.read_code(byte_length=1)
        instruction = code_to_instruction(code)

        while not isinstance(instruction, Halt):
            instruction.fetch(self)
            instruction.decode(self)
            instruction.execution(self)
            instruction.memory(self)
            instruction.write_back(self)
            instruction.pc_update(self)

            # read next instruction and decode
            code = self.read_code(byte_length=1)
            instruction = code_to_instruction(code)

    def run_assembly(self, program: str, start_memory: int = 0):
        bytecode = program_to_bytecode(program, base_memory=start_memory)
        self.load_byte_code(bytecode, start_memory)
        self.run()
        return self


def check_bytecode(code: str):
    pattern = r"\[offset\+(.*?)\]"
    label_names = re.findall(pattern, code)
    for label in label_names:
        code = code.replace(f"[offset+{label}]", num_to_hex(0))
    assert len(code) % 2 == 0, f"length of {code} is not even"


def program_to_bytecode(program: str, base_memory: int = None) -> Tuple[str, Dict]:
    """Convert program to bytecode"""
    bytecode = ""
    sentences = program.split("\n")
    offset = 0
    offset_map = {}
    for sen in sentences:
        uncomment_sen = sen.split("#")[0].strip()
        if not uncomment_sen.strip():
            continue

        if ":" in uncomment_sen:
            label = uncomment_sen.split(":")[0].strip()
            offset_map[label] = offset
            continue

        inst_data = uncomment_sen.split(maxsplit=1)
        if len(inst_data) == 1:
            inst, data = inst_data[0], ""
        else:
            inst, data = inst_data
        inst_obj: Instruction = name_to_instruction(inst)

        inst_code = inst_obj.to_byte_code(data.strip())
        offset += inst_obj.byte_length
        bytecode += inst_code
        check_bytecode(bytecode)

    if base_memory is None:
        return bytecode, offset_map
    else:
        for label, offset in offset_map.items():
            offset_value = base_memory + offset
            bytecode = bytecode.replace(f"[offset+{label}]", num_to_hex(offset_value))
        return bytecode


def bytecode_to_instruction(bytecode: str) -> str:
    program = ""
    pos = 0
    while pos < len(bytecode):
        code = bytecode[pos: pos + 2]
        inst = code_to_instruction(code)
        length = inst.byte_length
        full_code = bytecode[pos: pos + length * 2]
        val = inst.byte_to_assembly(full_code)
        program += val + "\n"
        pos += length * 2
        pass
    return program


def test_opq():
    p = Processor(num_registers=6, memory_size=1024)
    func_map = {
        0: lambda x, y: x + y,
        1: lambda x, y: x - y,
        2: lambda x, y: x & y,
        3: lambda x, y: x ^ y,
    }
    rA = [1, 3, 5, 7]
    rB = [2, 4, 6, 8]
    op_codes = [0, 1, 2, 3]
    ops = [func_map[x] for x in op_codes]
    value = [op(x, y) for x, y, op in zip(rA, rB, ops)]
    byte_code = [
        Opq.code + str(op_code) + str(a) + str(b) + Halt.code
        for op_code, a, b in zip(op_codes, rA, rB)
    ]

    for code, a, b, val in zip(byte_code, rA, rB, value):
        p.load_byte_code(code)
        p.set_register(code[2], a)
        p.set_register(code[3], b)
        p.run()
        result = p.read_register(b)
        assert result == val, f"{result} != {val} for {code}"

    print("Pass opq test!")


def test_rrmovq():
    p = Processor(num_registers=6, memory_size=1024)
    rA = [1, 3, 5, 7]
    rB = [2, 4, 6, 8]
    byte_code = [
        RRmovq.code + str(a) + str(b) + Halt.code
        for a, b in zip(rA, rB)
    ]

    for code, a, b in zip(byte_code, rA, rB):
        p.load_byte_code(code)
        p.set_register(code[2], a)
        p.run()
        result = p.read_register(b)
        assert result == a, f"{result} != {a} for {code}"

    print("Pass rrmovq test!")


def test_irmovq():
    p = Processor(num_registers=6, memory_size=1024)
    value = [random.randint(0, 1000000) for _ in range(4)]
    register = [2, 4, 6, 8]
    byte_code = [
        IRmovq.code + "F" + str(x) + num_to_hex(y) + Halt.code
        for x, y in zip(register, value)
    ]

    for code, val in zip(byte_code, value):
        p.load_byte_code(code)
        p.run()
        result = p.read_register(code[3])
        assert result == val, f"{result} != {val} for {code}"

    print("Pass irmovq test!")


def test_rmmovq():
    p = Processor(num_registers=6, memory_size=1024)
    addrs = [random.randint(0, 127) for _ in range(4)]
    rA = [1, 3, 5, 7]
    rB = [2, 4, 6, 8]
    byte_code = [
        RMmovq.code + str(a) + str(b) + num_to_hex(addr) + Halt.code
        for a, b, addr in zip(rA, rB, addrs)
    ]
    for code, addr, a, b in zip(byte_code, addrs, rA, rB):
        p.load_byte_code(code)
        p.set_register(code[2], a)
        p.set_register(code[3], b)
        p.run()
        result = hex_to_num(p.memory[addr + b])
        assert result == a, f"{result} != {a} for {code}"
    print("Pass rmmovq test!")


def test_mrmovq():
    p = Processor(num_registers=6, memory_size=1024)
    addrs = [random.randint(0, 1024) for _ in range(4)]
    rA = [1, 3, 5, 7]
    rB = [2, 4, 6, 8]
    byte_code = [
        MRmovq.code + str(a) + str(b) + num_to_hex(addr) + Halt.code
        for a, b, addr in zip(rA, rB, addrs)
    ]
    for code, addr, a, b in zip(byte_code, addrs, rA, rB):
        p.load_byte_code(code)
        p.memory[addr + b] = a
        p.set_register(code[2], a)
        p.set_register(code[3], b)
        p.run()
        result = p.read_register(a)
        assert result == a, f"{result} != {a} for {code}"
    print("Pass mrmovq test!")


def test_pushq():
    p = Processor(num_registers=6, memory_size=1024)
    values = [random.randint(1, 9) for _ in range(4)]  # exclude rsp
    byte_code = [Pushq.code + str(x) + "F" + Halt.code for x in values]
    for code, val in zip(byte_code, values):
        p.load_byte_code(code)
        p.set_register(code[2], val)
        p.run()
        addr = p.read_register("rsp")
        hex_result = p.memory[addr]
        assert len(hex_result) == 16, f"length of {hex_result} != 16"
        result = hex_to_num(hex_result)
        assert result == val, f"{result} != {val} for {code}"
    print("Pass pushq test!")


def test_popq():
    p = Processor(num_registers=6, memory_size=1024)
    reg_idx = [random.randint(1, 9) for _ in range(4)]  # exclude rsp
    byte_code = [Popq.code + str(x) + "F" + Halt.code for x in reg_idx]
    for code, reg_id in zip(byte_code, reg_idx):
        p.load_byte_code(code)
        addr = random.randint(900, 1023)
        value = random.randint(0, 1000000)
        p.set_register("rsp", addr)
        p.memory[addr] = value
        p.run()

        update_addr = p.read_register("rsp")
        mem_value = p.memory[update_addr - p.byte_length]
        result = p.read_register(reg_id)
        assert hex_to_num(mem_value) == result, f"{mem_value} != {result} for {code}"
        assert hex_to_num(mem_value) == value, f"{mem_value} != {value} for {code}"
    print("Pass popq test!")


def test_call():
    p = Processor(num_registers=6, memory_size=1024)
    values = [random.randint(0, 1023) for _ in range(4)]
    byte_code = [Call.code + num_to_hex(x) + Halt.code for x in values]
    for code, addr in zip(byte_code, values):
        p.load_byte_code(code)
        prev_rsp = p.read_register("rsp")
        p.run_one_step()
        after_rsp = p.read_register("rsp")
        assert prev_rsp - p.byte_length == after_rsp, f"{prev_rsp} != {after_rsp} for {code}"
        assert p.pc == addr, f"{p.pc} != {addr} for {code}"
    print("Pass call test!")


def test_ret():
    p = Processor(num_registers=6, memory_size=1024)
    addrs = [random.randint(0, 1023) for _ in range(4)]
    byte_code = [Ret.code + Halt.code for _ in addrs]
    for code, addr in zip(byte_code, addrs):
        p.load_byte_code(code)
        mem_value = random.randint(0, 1000000)
        p.set_register("rsp", addr)
        prev_rsp = p.read_register("rsp")
        p.memory[addr] = mem_value
        p.run_one_step()
        after_rsp = p.read_register("rsp")
        assert prev_rsp + p.byte_length == after_rsp, f"{prev_rsp} != {after_rsp}"
        assert p.pc == mem_value, f"{p.pc} != {mem_value}"
    print("Pass ret test!")


def test_instructions():
    test_opq()
    test_rrmovq()
    test_irmovq()
    test_rmmovq()
    test_mrmovq()
    test_pushq()
    test_popq()
    test_call()
    test_ret()


def test_sum_to10():
    program = r"""
    irmovq $0, %rax
    irmovq $0, %rbx
    irmovq $10, %rcx
    irmovq $1, %rdx

loop:
    subq %rbx, %rcx
    jg done

    irmovq $10, %rcx
    addq %rbx, %rax
    addq %rdx, %rbx

    jmp loop

done:
    halt
"""
    p = Processor(num_registers=6, memory_size=1024)
    p.run_assembly(program)

    sum_val = p.read_register("rax")
    assert sum_val == 45, f"{sum_val} != 45"
    print("Pass test_sum_to10!")


def test_processor_execution():
    program = r"""
    irmovq $9, %rdx
    irmovq $21, %rbx

    subq %rdx, %rbx
    je done
    irmovq $1024, %rsp

    pushq %rdx
    popq %rax

    call proc

done:
    halt

proc:
    irmovq $13, %rdx
    ret
"""
    start_memory = 20
    p = Processor(num_registers=6, memory_size=1024)
    p.run_assembly(program, start_memory=start_memory)

    assert p.read_register("rbx") == -12, f"{p.read_register('rbx')} != -12"
    assert p.read_register("rax") == 9, f"{p.read_register('rax')} != 9"
    assert p.read_register("rdx") == 13, f"{p.read_register('rdx')} != 13"
    assert p.memory[:start_memory] == "".join(["00" for _ in range(start_memory)])
    print("Pass test_processor_execution!")


def test_byte_assembly_convert():
    assembly = [
        r"irmovq $9, %rdx",
        r"rrmovq %rdx, %rbx",
        r"rmmovq %rdx, 123(%rbx)",
        r"mrmovq 126(%rbx), %rax",
        r"addq %rdx, %rbx",
        r"subq %rdx, %rbx",
        r"pushq %rdx",
        r"popq %rax",
        r"ret",
        r"""
irmovq $9, %rdx
subq %rdx, %rbx
pushq %rdx
popq %rax
ret
""".strip()
    ]
    for x in assembly:
        byte_code = program_to_bytecode(x, base_memory=0)
        back_x = bytecode_to_instruction(byte_code).strip()
        assert back_x == x, f"{back_x} != {x}"


if __name__ == "__main__":
    test_instructions()
    test_sum_to10()
    test_processor_execution()
    test_byte_assembly_convert()
