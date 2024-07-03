// generated from rosidl_typesupport_introspection_cpp/resource/idl__type_support.cpp.em
// with input from kobuki_ros_interfaces:msg/DigitalOutput.idl
// generated code does not contain a copyright notice

#include "array"
#include "cstddef"
#include "string"
#include "vector"
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_interface/macros.h"
#include "kobuki_ros_interfaces/msg/detail/digital_output__struct.hpp"
#include "rosidl_typesupport_introspection_cpp/field_types.hpp"
#include "rosidl_typesupport_introspection_cpp/identifier.hpp"
#include "rosidl_typesupport_introspection_cpp/message_introspection.hpp"
#include "rosidl_typesupport_introspection_cpp/message_type_support_decl.hpp"
#include "rosidl_typesupport_introspection_cpp/visibility_control.h"

namespace kobuki_ros_interfaces
{

namespace msg
{

namespace rosidl_typesupport_introspection_cpp
{

void DigitalOutput_init_function(
  void * message_memory, rosidl_runtime_cpp::MessageInitialization _init)
{
  new (message_memory) kobuki_ros_interfaces::msg::DigitalOutput(_init);
}

void DigitalOutput_fini_function(void * message_memory)
{
  auto typed_message = static_cast<kobuki_ros_interfaces::msg::DigitalOutput *>(message_memory);
  typed_message->~DigitalOutput();
}

size_t size_function__DigitalOutput__values(const void * untyped_member)
{
  (void)untyped_member;
  return 4;
}

const void * get_const_function__DigitalOutput__values(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::array<bool, 4> *>(untyped_member);
  return &member[index];
}

void * get_function__DigitalOutput__values(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::array<bool, 4> *>(untyped_member);
  return &member[index];
}

void fetch_function__DigitalOutput__values(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const bool *>(
    get_const_function__DigitalOutput__values(untyped_member, index));
  auto & value = *reinterpret_cast<bool *>(untyped_value);
  value = item;
}

void assign_function__DigitalOutput__values(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<bool *>(
    get_function__DigitalOutput__values(untyped_member, index));
  const auto & value = *reinterpret_cast<const bool *>(untyped_value);
  item = value;
}

size_t size_function__DigitalOutput__mask(const void * untyped_member)
{
  (void)untyped_member;
  return 4;
}

const void * get_const_function__DigitalOutput__mask(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::array<bool, 4> *>(untyped_member);
  return &member[index];
}

void * get_function__DigitalOutput__mask(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::array<bool, 4> *>(untyped_member);
  return &member[index];
}

void fetch_function__DigitalOutput__mask(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const bool *>(
    get_const_function__DigitalOutput__mask(untyped_member, index));
  auto & value = *reinterpret_cast<bool *>(untyped_value);
  value = item;
}

void assign_function__DigitalOutput__mask(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<bool *>(
    get_function__DigitalOutput__mask(untyped_member, index));
  const auto & value = *reinterpret_cast<const bool *>(untyped_value);
  item = value;
}

static const ::rosidl_typesupport_introspection_cpp::MessageMember DigitalOutput_message_member_array[2] = {
  {
    "values",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_BOOLEAN,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    4,  // array size
    false,  // is upper bound
    offsetof(kobuki_ros_interfaces::msg::DigitalOutput, values),  // bytes offset in struct
    nullptr,  // default value
    size_function__DigitalOutput__values,  // size() function pointer
    get_const_function__DigitalOutput__values,  // get_const(index) function pointer
    get_function__DigitalOutput__values,  // get(index) function pointer
    fetch_function__DigitalOutput__values,  // fetch(index, &value) function pointer
    assign_function__DigitalOutput__values,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "mask",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_BOOLEAN,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    4,  // array size
    false,  // is upper bound
    offsetof(kobuki_ros_interfaces::msg::DigitalOutput, mask),  // bytes offset in struct
    nullptr,  // default value
    size_function__DigitalOutput__mask,  // size() function pointer
    get_const_function__DigitalOutput__mask,  // get_const(index) function pointer
    get_function__DigitalOutput__mask,  // get(index) function pointer
    fetch_function__DigitalOutput__mask,  // fetch(index, &value) function pointer
    assign_function__DigitalOutput__mask,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  }
};

static const ::rosidl_typesupport_introspection_cpp::MessageMembers DigitalOutput_message_members = {
  "kobuki_ros_interfaces::msg",  // message namespace
  "DigitalOutput",  // message name
  2,  // number of fields
  sizeof(kobuki_ros_interfaces::msg::DigitalOutput),
  DigitalOutput_message_member_array,  // message members
  DigitalOutput_init_function,  // function to initialize message memory (memory has to be allocated)
  DigitalOutput_fini_function  // function to terminate message instance (will not free memory)
};

static const rosidl_message_type_support_t DigitalOutput_message_type_support_handle = {
  ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  &DigitalOutput_message_members,
  get_message_typesupport_handle_function,
};

}  // namespace rosidl_typesupport_introspection_cpp

}  // namespace msg

}  // namespace kobuki_ros_interfaces


namespace rosidl_typesupport_introspection_cpp
{

template<>
ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
get_message_type_support_handle<kobuki_ros_interfaces::msg::DigitalOutput>()
{
  return &::kobuki_ros_interfaces::msg::rosidl_typesupport_introspection_cpp::DigitalOutput_message_type_support_handle;
}

}  // namespace rosidl_typesupport_introspection_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, kobuki_ros_interfaces, msg, DigitalOutput)() {
  return &::kobuki_ros_interfaces::msg::rosidl_typesupport_introspection_cpp::DigitalOutput_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif
